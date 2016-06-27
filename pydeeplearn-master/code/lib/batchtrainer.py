""" A neural network trainer with various options: rmsprop, nerterov, momentum etc.
This trainer is used for multiple types of neural nets: dbns and cnns and it is designed
to be adaptable to other types of networks as well.
"""

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

import numpy as np
import theano
from theano import tensor as T
import common

import debug

DEBUG = False
theanoFloat = theano.config.floatX

class BatchTrainer(object):
  """
    Abstract class used to define updates on the parameters of neural networks
    during training.

    Subclasses must call the constructor of this class in their constructors, and
    have to define their cost function using a method called 'cost'.

    Supports momentum updates and nesterov updates, both with rmsprop
    or without (see the TrainingOptions class for more details in the available
    training options. Also supports L1 and L2 weight decay.
  """

  def __init__(self, params, weights, training_options):
    self.params = params
    self.training_options = training_options
    self.weights = weights if weights else []

    # Required for momentum and rmsprop
    self.oldUpdates = []
    self.oldMeanSquares = []
    for param in params:
      oldDParam = theano.shared(value=np.zeros(shape=param.shape.eval(),
                                              dtype=theanoFloat),
                                name='oldDParam')

      self.oldUpdates += [oldDParam]
      oldMeanSquare = theano.shared(value=np.zeros(shape=param.shape.eval(),
                                              dtype=theanoFloat),
                                name='oldMeanSquare')

      self.oldMeanSquares += [oldMeanSquare]


  def trainFixedEpochs(self, x, y, data, labels, maxEpochs):
    training_options = self.training_options
    trainModel = self._makeTrainFunction(x, y, data, labels)
    epochTrainingErrors = []
    nrMiniBatchesTrain = max(data.shape.eval()[1] / training_options.miniBatchSize, 1)

    best_training_error = np.inf
    bestWeights = None
    bestBiases = None
    bestEpoch = 0

    save_best_weights = training_options.save_best_weights

    try:
      for epoch in xrange(maxEpochs):
        print "epoch " + str(epoch)
        momentum = training_options.momentumForEpochFunction(training_options.momentumMax, epoch)
        sum_error = 0.0
        for batchNr in xrange(nrMiniBatchesTrain):
          trainError = trainModel(batchNr, momentum) / training_options.miniBatchSize
          sum_error += trainError

        mean_error = sum_error / nrMiniBatchesTrain
        if save_best_weights:
          if mean_error < best_training_error:
            best_training_error = mean_error
            # Save the weights which are the best ones
            bestWeights = self.weights
            bestBiases = self.biases
            bestEpoch = epoch

        print "training error " + str(mean_error)
        epochTrainingErrors += [mean_error]
    except KeyboardInterrupt:
      print "you have interrupted training"
      print "we will continue testing with the state of the network as it is"

    if save_best_weights:
      if bestWeights is not None and bestBiases is not None:
        self.weights = bestWeights
        self.biases = bestBiases

    print "number of epochs"
    print epoch + 1

  def trainWithValidation(self, x, y, data, labels, validationData, validationLabels,
      classificationCost, maxEpochs, validation_criteria):
    if validation_criteria == "patience":
      self._trainModelPatience(x, y, data, labels, validationData, validationLabels, classificationCost, maxEpochs)
    elif validation_criteria == "consecutive_decrease":
      self._trainLoopWithValidation(x, y, data, labels, validationData, validationLabels, classificationCost, maxEpochs)
    else:
      raise Exception("unknown validation validation_criteria: " + str(validation_criteria))

  def _trainLoopWithValidation(self, x, y, data, labels, validationData, validationLabels,
      classificationCost, maxEpochs):
    lastValidationError = np.inf
    consecutive_decrease_error_count = 0.0
    epoch = 0
    training_options = self.training_options
    save_best_weights = training_options.save_best_weights

    miniBatchSize = training_options.miniBatchSize
    nrMiniBatchesTrain = max(data.shape.eval()[0] / miniBatchSize, 1)
    miniBatchValidateSize = min(validationData.shape.eval()[0], miniBatchSize * 10)
    nrMiniBatchesValidate = max(validationData.shape.eval()[0] / miniBatchValidateSize, 1)

    trainModel = self._makeTrainFunction(x, y, data, labels)
    validateModel = self._makeValidateModelFunction(
        x, y, validationData, validationLabels, classificationCost, miniBatchValidateSize)
    trainNoDropout = self._makeValidateModelFunction(
        x, y, data, labels, classificationCost, miniBatchSize)

    validationErrors = []
    trainingErrors = []
    trainingErrorsNoDropout = []

    bestValidationError = np.inf
    bestWeights = None
    bestBiases = None
    bestEpoch = 0

    try:
      while epoch < maxEpochs and consecutive_decrease_error_count < 8:
        print "epoch " + str(epoch)

        momentum = self.training_options.momentumForEpochFunction(training_options.momentumMax, epoch)
        sumErrors = 0.0
        sumErrorsNoDropout = 0.0
        for batchNr in xrange(nrMiniBatchesTrain):
          sumErrors += trainModel(batchNr, momentum) / miniBatchSize
          sumErrorsNoDropout += trainNoDropout(batchNr) / miniBatchSize

        trainingErrors += [sumErrors / nrMiniBatchesTrain]
        trainingErrorsNoDropout += [sumErrorsNoDropout / nrMiniBatchesTrain]

        meanValidations = map(validateModel, xrange(nrMiniBatchesValidate))
        meanValidationError = sum(meanValidations) / len(meanValidations)
        validationErrors += [meanValidationError]

        if save_best_weights:
          if meanValidationError < bestValidationError:
            bestValidationError = meanValidationError
            # Save the weights which are the best ones
            bestWeights = self.weights
            bestBiases = self.biases
            bestEpoch = epoch

        consecutive_decrease_error_count = consecutive_decrease_error_count + 1 if meanValidationError > lastValidationError else 0
        lastValidationError = meanValidationError
        epoch += 1

    except KeyboardInterrupt:
      print "you have interrupted training"
      print "we will continue testing with the state of the network as it is"

    # TODO: flag for plotting
    common.plotTrainingAndValidationErros(trainingErrors, validationErrors)
    common.plotTrainingAndValidationErros(trainingErrorsNoDropout, validationErrors)

    print "number of epochs"
    print epoch + 1


  def _trainModelPatience(self, x, y, data, labels, validationData, validationLabels,
      classificationCost, maxEpochs):
    training_options = self.training_options
    save_best_weights = training_options.save_best_weights

    miniBatchSize = training_options.miniBatchSize
    nrMiniBatchesTrain = max(data.shape.eval()[0] / miniBatchSize, 1)
    miniBatchValidateSize = min(validationData.shape.eval()[0], miniBatchSize * 10)
    nrMiniBatchesValidate = max(validationData.shape.eval()[0] / miniBatchValidateSize, 1)

    trainModel = self._makeTrainFunction(x, y, data, labels)
    validateModel = self._makeValidateModelFunction(
        x, y, validationData, validationLabels, classificationCost, miniBatchValidateSize)
    trainNoDropout = self._makeValidateModelFunction(
        x, y, data, labels, classificationCost, miniBatchSize)

    epoch = 0
    doneTraining = False
    patience = 10 * nrMiniBatchesTrain  # do at least 10 passes trough the data no matter what
    patienceIncrease = 2  # Increase our patience up to patience * patienceIncrease

    bestValidationError = np.inf
    bestWeights = None
    bestBiases = None
    bestEpoch = 0

    validationErrors = []
    trainingErrors = []
    trainingErrorNoDropout = []

    try:
      while (epoch < maxEpochs) and not doneTraining:
        # Train the net with all data
        print "epoch " + str(epoch)
        momentum = training_options.momentumForEpochFunction(training_options.momentumMax, epoch)

        for batchNr in xrange(nrMiniBatchesTrain):
          iteration = epoch * nrMiniBatchesTrain  + batchNr
          trainingErrorBatch = trainModel(batchNr, momentum) / training_options.miniBatchSize

          meanValidations = map(validateModel, xrange(nrMiniBatchesValidate))
          meanValidationError = sum(meanValidations) / len(meanValidations)

          if meanValidationError < bestValidationError:
            print "increasing patience, still improving during training..."
            patience = max(patience, iteration * patienceIncrease)
            bestValidationError = meanValidationError
            if save_best_weights:
              # Save the weights which are the best ones
              bestWeights = self.weights
              bestBiases = self.biases
              bestEpoch = epoch

        validationErrors += [meanValidationError]
        trainingErrors += [trainingErrorBatch]
        trainingErrorNoDropout +=  [trainNoDropout(batchNr)]

        if patience <= iteration:
          doneTraining = True

        epoch += 1
    except KeyboardInterrupt:
      print "you have interrupted training"
      print "we will continue testing with the state of the network as it is"

    # TODO: double check
    if save_best_weights:
      if bestWeights is not None and bestBiases is not None:
        self.weights = bestWeights
        self.biases = bestBiases

    common. plotTrainingAndValidationErros(trainingErrors, validationErrors)
    common.plotTrainingAndValidationErros(trainingErrorNoDropout, validationErrors)

    print "number of epochs"
    print epoch


  # TODO: document cost
  def _makeValidateModelFunction(self, x, y, data, labels, cost, miniBatchSize):
    miniBatchIndex = T.lscalar()

    return theano.function(
        inputs=[miniBatchIndex],
        outputs=T.mean(cost(y)),
        givens={
            x: data[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
            y: labels[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize]})


  def _makeTrainFunction(self, x, y, data, labels):
    error = T.sum(self.cost(y))
    training_options = self.training_options

    for w in self.weights:
      error += training_options.weightDecayL1 * T.sum(abs(w))
      error += training_options.weightDecayL2 * T.sum(w ** 2)

    miniBatchIndex = T.lscalar()
    momentum = T.fscalar()

    if DEBUG:
      mode = theano.compile.MonitorMode(post_func=debug.detect_nan).excluding(
                                        'local_elemwise_fusion', 'inplace')
    else:
      mode = None

    if training_options.nesterov:
      preDeltaUpdates, updates = self._buildUpdatesNesterov(error, training_options, momentum)
      updateParamsWithMomentum = theano.function(
          inputs=[momentum],
          outputs=[],
          updates=preDeltaUpdates,
          mode=mode)

      updateParamsWithGradient = theano.function(
          inputs =[miniBatchIndex, momentum],
          outputs=error,
          updates=updates,
          givens={
              x: data[miniBatchIndex * training_options.miniBatchSize:(miniBatchIndex + 1) * training_options.miniBatchSize],
              y: labels[miniBatchIndex * training_options.miniBatchSize:(miniBatchIndex + 1) * training_options.miniBatchSize]},
          mode=mode)

      def trainModel(miniBatchIndex, momentum):
        updateParamsWithMomentum(momentum)
        return updateParamsWithGradient(miniBatchIndex, momentum)

    else:
      updates = self._buildUpdatesSimpleMomentum(error, training_options, momentum)
      trainModel = theano.function(
            inputs=[miniBatchIndex, momentum],
            outputs=error,
            updates=updates,
            mode=mode,
            givens={
                x: data[miniBatchIndex * training_options.miniBatchSize:(miniBatchIndex + 1) * training_options.miniBatchSize],
                y: labels[miniBatchIndex * training_options.miniBatchSize:(miniBatchIndex + 1) * training_options.miniBatchSize]})

    # returns the function that trains the model
    return trainModel

  def _buildUpdatesNesterov(self, error, training_options, momentum):
    if training_options.momentumFactorForLearningRate:
      lrFactor =  np.float32(1.0) - momentum
    else:
      lrFactor = np.float32(1.0)

    preDeltaUpdates = []
    for param, oldUpdate in zip(self.params, self.oldUpdates):
      preDeltaUpdates.append((param, param + momentum * oldUpdate))

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    deltaParams = T.grad(error, self.params)
    updates = []
    parametersTuples = zip(self.params,
                           deltaParams,
                           self.oldUpdates,
                           self.oldMeanSquares)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      if training_options.rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate = - lrFactor * training_options.batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate = - lrFactor * training_options.batchLearningRate * delta

      newParam = param + paramUpdate

      updates.append((param, newParam))
      updates.append((oldUpdate, momentum * oldUpdate + paramUpdate))

    return preDeltaUpdates, updates

  def _buildUpdatesSimpleMomentum(self, error, training_options, momentum):
    if training_options.momentumFactorForLearningRate:
      lrFactor = np.float32(1.0) - momentum
    else:
      lrFactor = np.float32(1.0)

    deltaParams = T.grad(error, self.params)
    updates = []
    parametersTuples = zip(self.params,
                           deltaParams,
                           self.oldUpdates,
                           self.oldMeanSquares)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      paramUpdate = momentum * oldUpdate
      if training_options.rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate += - lrFactor * training_options.batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate += - lrFactor * training_options.batchLearningRate * delta

      newParam = param + paramUpdate

      updates.append((param, newParam))
      updates.append((oldUpdate, paramUpdate))

    return updates
