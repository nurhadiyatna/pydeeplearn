""" Defines a training options class as a holder for options that can be passed
for training a neural network.
"""

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

import numpy as np
# TODO: move from common here
import common

class TrainingOptions(object):

  def __init__(self, miniBatchSize,
        learningRate,
        momentumMax=0.0,
        rmsprop=False,
        weightDecayL1=0.0,
        weightDecayL2=0.0,
        nesterovMomentum=False,
        save_best_weights=False,
        momentumForEpochFunction=common.getMomentumForEpochLinearIncrease,
        momentumFactorForLearningRate=False):
    self.miniBatchSize = miniBatchSize
    self.learningRate = learningRate
    self.momentumMax = np.float32(momentumMax)
    self.rmsprop = rmsprop
    self.weightDecayL1 = weightDecayL1
    self.weightDecayL2 = weightDecayL2
    self.nesterov = nesterovMomentum
    self.momentumFactorForLearningRate = momentumFactorForLearningRate
    self.momentumForEpochFunction = momentumForEpochFunction
    self.batchLearningRate = np.float32(learningRate / miniBatchSize)
    self.save_best_weights = save_best_weights
