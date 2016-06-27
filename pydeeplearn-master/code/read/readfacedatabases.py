"""The aim of this script is to read the face databases used for experiments """

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import glob
import cPickle as pickle
import os
import cv2
import facedetection
import fnmatch
import csv

import matplotlib.image as io
from skimage.transform import resize

from lib.common import *

SMALL_SIZE = ((40, 30))

PATH = "/home/aela/uni/project/"

def equalizeImgGlobal(x):
  return cv2.equalizeHist(x)

def equalizeCLAHE(x):
  # Contrast Limited Adaptive Histogram Equalization
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
  return clahe.apply(x)


def equalizeFromFloatGlobal(x, reshapeSize=SMALL_SIZE):
  x = x * 255
  x = np.asarray(x, dtype='uint8')
  y = x.reshape(reshapeSize)
  y =  equalizeImgGlobal(y).reshape(-1)
  return y / 255.0


def cropFromFloat(x):
  x = x * 255
  x = np.asarray(x, dtype='uint8')
  # y = x.reshape(SMALL_SIZE)
  y = facedetection.cropFace(x, rescaleForReconigtion=1)
  if y is None:
    return None
  return y.reshape(SMALL_SIZE)


def equalizeFromFloatCLAHE(x, reshapeSize=SMALL_SIZE):
  x = x * 255
  x = np.asarray(x, dtype='uint8')
  y = x.reshape(reshapeSize)
  y =  equalizeCLAHE(y).reshape(-1)
  return y / 255.0



def equalizePIE(train=True):
  imgs, labels = readMultiPIE(train)
  imgs = np.array(map(lambda x: equalizeFromFloatGlobal(x), imgs))

  return imgs, labels

def equalizeKanade(big=False):
  data, labels = 6(big=big, equalize=False)

  if big:
      fileName = 'equalized_kanade_big.pickle'
  else:
      fileName = 'equalized_kanade_small.pickle'


  data = np.array(map(lambda x: equalizeFromFloatCLAHE(x), data))

  with open(fileName, "wb") as f:
    pickle.dump(data, f)
    pickle.dump(labels, f)

def readMultiPIE(
    show=False, equalize=False, vectorizeLabels=True, globalPath=PATH, train=True):
  path = os.path.join(globalPath, "Multi-PIE_Aligned/A_MultiPIE.mat")

  if equalize:
    print "equalizing multi pie data"
  mat = scipy.io.loadmat(path)
  data = mat['a_multipie']
  # For all the subjects
  imgs = []
  labels = []
  for subject in xrange(147):
    for pose in xrange(5):
      for expression in xrange(6): # ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
        for illumination in xrange(5):
            image = np.squeeze(data[subject,pose,expression,illumination,:])

            if equalize:
              image = equalizeFromFloatGlobal(image)

            image = image.reshape(30,40).T
            if show:
              plt.imshow(image, cmap=plt.cm.gray)
              plt.show()
            imgs += [image.reshape(-1)]
            labels += [expression]

  if vectorizeLabels:
    labels = labelsToVectors(labels, 6)
  else:
    labels = np.array(labels)

  nrTrain = len(imgs) * 9 / 10

  if train:
    return np.array(imgs)[0:nrTrain], labels[0:nrTrain]
  else:
    return np.array(imgs)[nrTrain:], labels[nrTrain:]

def readMultiPieDifferentIlluminations(illuminationTrain, show=False, equalize=False, globalPath=PATH):
  path = os.path.join(globalPath, "Multi-PIE_Aligned/A_MultiPIE.mat")

  mat = scipy.io.loadmat(path)
  data = mat['a_multipie']
  # For all the subjects
  imgsTrain = []
  labelsTrain = []

  imgsTest = []
  labelsTest = []

  for subject in xrange(147):
    for pose in xrange(5):
      for expression in xrange(6): # ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
        for illumination in xrange(5):
            image = np.squeeze(data[subject,pose,expression,illumination,:])

            if equalize:
              image = equalizeFromFloatGlobal(image)

            image = image.reshape(30,40).T
            if illumination in illuminationTrain:
              imgsTrain += [image.reshape(-1)]
              labelsTrain += [expression]

              if show:
                plt.imshow(image, cmap=plt.cm.gray)
                plt.show()

            else:
              imgsTest += [image.reshape(-1)]
              labelsTest += [expression]

  # Let us shuffle some things
  imgsTrain = np.array(imgsTrain)
  labelsTrain = np.array(labelsTrain)

  imgsTest = np.array(imgsTest)
  labelsTest = np.array(labelsTest)

  imgsTrain, labelsTrain = shuffle(imgsTrain, labelsTrain)
  imgsTest, labelsTest = shuffle(imgsTest, labelsTest)

  return (imgsTrain, labelsToVectors(labelsTrain, 6),
          imgsTest,  labelsToVectors(labelsTest, 6))

def readMultiPieDifferentSubjects(subjectsTrain, show=False, equalize=False, globalPath=PATH):
  path = os.path.join(globalPath, "Multi-PIE_Aligned/A_MultiPIE.mat")

  mat = scipy.io.loadmat(path)
  data = mat['a_multipie']
  # For all the subjects
  imgsTrain = []
  labelsTrain = []

  imgsTest = []
  labelsTest = []

  for subject in xrange(147):
    for pose in xrange(5):
      for expression in xrange(6): # ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
        for illumination in xrange(5):
            image = np.squeeze(data[subject,pose,expression,illumination,:])

            if equalize:
              image = equalizeFromFloatGlobal(image)

            image = image.reshape(30,40).T
            if subject in subjectsTrain:
              imgsTrain += [image.reshape(-1)]
              labelsTrain += [expression]

              if show:
                plt.imshow(image, cmap=plt.cm.gray)
                plt.show()

            else:
              imgsTest += [image.reshape(-1)]
              labelsTest += [expression]

  # Let us shuffle some things
  imgsTrain = np.array(imgsTrain)
  labelsTrain = np.array(labelsTrain)

  imgsTest = np.array(imgsTest)
  labelsTest = np.array(labelsTest)

  imgsTrain, labelsTrain = shuffle(imgsTrain, labelsTrain)
  imgsTest, labelsTest = shuffle(imgsTest, labelsTest)

  return (imgsTrain, labelsToVectors(labelsTrain, 6),
          imgsTest,  labelsToVectors(labelsTest, 6))

def readMultiPieDifferentPoses(posesTrain, show=False, equalize=False, globalPath=PATH):
  path = os.path.join(globalPath, "Multi-PIE_Aligned/A_MultiPIE.mat")

  mat = scipy.io.loadmat(path)
  data = mat['a_multipie']
  # For all the subjects
  imgsTrain = []
  labelsTrain = []

  imgsTest = []
  labelsTest = []

  for subject in xrange(147):
    for pose in xrange(5):
      for expression in xrange(6): # ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
        for illumination in xrange(5):
            image = np.squeeze(data[subject,pose,expression,illumination,:])
            if equalize:
              image = equalizeFromFloatGlobal(image)

            image = image.reshape(30,40).T

            if pose in posesTrain:
              imgsTrain += [image.reshape(-1)]
              labelsTrain += [expression]
              # Only show the faces for this pose
              if show and expression == 5 and subject == 5:
                plt.imshow(image, cmap=plt.cm.gray)
                plt.axis('off')
                plt.show()
            else:
              imgsTest += [image.reshape(-1)]
              labelsTest += [expression]

  # Let us shuffle some things
  imgsTrain = np.array(imgsTrain)
  labelsTrain = np.array(labelsTrain)

  imgsTest = np.array(imgsTest)
  labelsTest = np.array(labelsTest)

  imgsTrain, labelsTrain = shuffle(imgsTrain, labelsTrain)
  imgsTest, labelsTest = shuffle(imgsTest, labelsTest)

  return (imgsTrain, labelsToVectors(labelsTrain, 6),
          imgsTest,  labelsToVectors(labelsTest, 6))



def makeMultiPiePlot():
  data, labels = readMultiPIE(False, False, vectorizeLabels=False, globalPath=PATH)

  data, labels  = shuffle(data, labels)
  finalPics = []
  for i in xrange(6):
    indices = labels == i
    finalPics += [data[indices][0].reshape(SMALL_SIZE)]

  img = np.hstack(tuple(finalPics))

  plt.imshow(img, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()


def makeMultiPosesPlot():
  finalPics = []
  for i in xrange(5):
    pics, _, _, _ = readMultiPieDifferentPoses([i], show=False, equalize=False)
    finalPics += [pics[0].reshape(SMALL_SIZE)]

  img = np.hstack(tuple(finalPics))

  plt.imshow(img, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()


def makeMultiIlluminationsPlot():
  finalPics = []
  for i in xrange(5):
    pics, _, _, _ = readMultiPieDifferentIlluminations([i], show=False, equalize=False)
    finalPics += [pics[0].reshape(SMALL_SIZE)]

  img = np.hstack(tuple(finalPics))

  plt.imshow(img, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

def readMultiPIESubjects(equalize=False, globalPath=PATH):
  path = os.path.join(globalPath, "Multi-PIE_Aligned/A_MultiPIE.mat")

  mat = scipy.io.loadmat(path)
  data = mat['a_multipie']
  subjectsToImgs = {}
  # For all the subjects

  for subject in xrange(147):
    subjectsToImgs[subject] = []
    for pose in xrange(5):
      for expression in xrange(6): # ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
        for illumination in xrange(5):
            image = np.squeeze(data[subject,pose,expression,illumination,:])

            if equalize:
              image = equalizeFromFloatGlobal(image)

            image = image.reshape(30,40).T
            image = image.reshape(-1)

            subjectsToImgs[subject] += [image]

  return subjectsToImgs


def readMultiPIEEmotions(equalize=False, globalPath=PATH):
  path = os.path.join(globalPath, "Multi-PIE_Aligned/A_MultiPIE.mat")

  mat = scipy.io.loadmat(path)
  data = mat['a_multipie']
  emotionToImages = {}
  # For all the subjects

  for expression in xrange(6):
    emotionToImages[expression] = []
  for subject in xrange(147):
    for pose in xrange(5):
      for expression in xrange(6): # ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
        for illumination in xrange(5):
            image = np.squeeze(data[subject,pose,expression,illumination,:])

            if equalize:
              image = equalizeFromFloatGlobal(image)

            image = image.reshape(30,40).T
            image = image.reshape(-1)

            emotionToImages[expression] += [image]

  return emotionToImages


def makeKanadeImages():
  data, labels = readKanade(vectorizeLabels=False)

  if 7 in labels:
    labels = labels -1

  print np.unique(labels)

  images = []
  for i in xrange(7):
    print i
    emotionData = data[labels == i]
    images += [emotionData[0].reshape(SMALL_SIZE)]

  images = np.hstack(images)

  plt.imshow(images, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

def makeMultiPieImagesForReport(globalPath=PATH):
  path = os.path.join(globalPath, "Multi-PIE_Aligned/A_MultiPIE.mat")
  mat = scipy.io.loadmat(path)
  data = mat['a_multipie']

  nrSubjects = 3

  total = []
  for x in xrange(nrSubjects):
    total += [[1] * 6]

  subjects = [0, 43, 140]

  for subject in xrange(147):
    print subject
    if subject not in subjects:
      continue

    index = subjects.index(subject)
    print "index", index

    for pose in xrange(5):
      if pose != 2 * index:
        continue
      for expression in xrange(6): # ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
        for illumination in xrange(5):

            if illumination != index:
              continue

            image = np.squeeze(data[subject,pose,expression,illumination,:])

            image = image.reshape(30,40).T
            total[index][expression] = image

    if index + 1 >= nrSubjects:
      break

  final = []
  for images in total:
    images = np.hstack(images)
    print images.shape
    final += [images]

  final = np.vstack(final)

  plt.imshow(final, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

def makeEqualizePics():
  # Read from multi PIE, Kanade, Jaffe and Aberdeen one image and then
  # plot it] PATH = '/home/aela/uni/project/Multi-PIE_Aligned/A_MultiPIE.mat'
  data, _ = readMultiPIE()
  pie = data[0]
  pie = pie.reshape(SMALL_SIZE)
  pieEq = equalizeFromFloatGlobal(pie)
  pieEq = pieEq.reshape(SMALL_SIZE)


  data, _ = readKanade()
  kanade = data[0]
  kanade = kanade.reshape(SMALL_SIZE)
  kanadeEq = equalizeFromFloatCLAHE(kanade)
  kanadeEq = kanadeEq.reshape(SMALL_SIZE)


  data = readJaffe(crop=True, detectFaces=True, equalize=False)
  jaffe = data[0]
  jaffe = jaffe.reshape(SMALL_SIZE)
  jaffeEq = equalizeFromFloatCLAHE(jaffe)
  jaffeEq = jaffeEq.reshape(SMALL_SIZE)

  data = readCroppedYale(False)
  yale = data[0]
  yale = yale.reshape(SMALL_SIZE)
  yaleEq = equalizeFromFloatCLAHE(yale)
  yaleEq = yaleEq.reshape(SMALL_SIZE)


  first = np.hstack((pie, kanade, jaffe, yale))
  second = np.hstack((pieEq, kanadeEq, jaffeEq, yaleEq))

  allPics = np.vstack((first, second))

  plt.imshow(allPics, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()


def facedetectionMultiPie():
  data, _ = readMultiPIE()

  data = map(lambda x: x.reshape(SMALL_SIZE), data)
  data = map(cropFromFloat, data)
  for d in data:
    if d is not None:
      plt.imshow(d, cmap=plt.cm.gray)
      plt.show()


def makeCrossDbPlot():
  dataKanade, labelsKanade = readKanade(vectorizeLabels=False, equalize=True)
  dataPie, labelsPie, _, _ = readMultiPieDifferentPoses([2], equalize=True)
  labelsPie = np.argmax(labelsPie, axis=1)

  if 7 in labelsKanade:
    labelsKanade = labelsKanade - 1

  dataKanade, labelsKanade = mapKanadeToPIELabels(dataKanade, labelsKanade)

  kanadePics = []
  piePics = []
  for i in xrange(6):
    print "at emotion ", i

    pie = dataPie[labelsPie == i]
    pie = pie[0]
    print "pie.shape"
    print pie.shape
    pie = pie.reshape(SMALL_SIZE)

    kanade = dataKanade[labelsKanade == i]
    kanade = kanade[0]
    kanade = kanade.reshape(SMALL_SIZE)
    kanadePics += [kanade]
    piePics += [pie]


  image1 = np.hstack(tuple(kanadePics))
  image2 = np.hstack(tuple(piePics))

  images = np.vstack((image1, image2))

  plt.imshow(images, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()


def readMultiPIEEmotionsPerSubject(equalize, globalPath=PATH):
  path = os.path.join(globalPath, "Multi-PIE_Aligned/A_MultiPIE.mat")

  mat = scipy.io.loadmat(path)
  data = mat['a_multipie']
  # For all the subjects
  subjectToEmotions = []

  for subject in xrange(147):
    emotionToImages = {}
    # initialize the dictionary for this subject
    for e in xrange(6):
      emotionToImages[e] = []

    for pose in xrange(5):
      for expression in xrange(6): # ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
        for illumination in xrange(5):
            image = np.squeeze(data[subject,pose,expression,illumination,:])
            if equalize:
              image = equalizeFromFloatGlobal(image)

            image = image.reshape(30,40).T
            image = image.reshape(-1)

            emotionToImages[expression] += [image]
    subjectToEmotions+= [emotionToImages]

  return subjectToEmotions

def readKanade(
  big=False, folds=None, equalize=False, vectorizeLabels=True, train=True):
  if not equalize:
    if big:
      files = glob.glob('kanade_150*.pickle')
    else:
      files = glob.glob('kanade_f*.pickle')

    if not folds:
      folds = range(1, 6)

    # Read the data from them. Sort out the files that do not have
    # the folds that we want
    # Do not reply on the order returned
    files = [ files[x -1] for x in folds]

    data = np.array([])
    labels = np.array([])

    dataFolds = []
    labelFolds = []
    for filename in files:
      with open(filename, "rb") as  f:
        # Sort out the labels from the data
        dataAndLabels = pickle.load(f)
        foldData = dataAndLabels[:, 0:-1]
        print "foldData.shape"
        print foldData.shape
        foldLabels = dataAndLabels[:,-1]
        dataFolds.append(foldData)

        foldLabels = np.array(map(int, foldLabels))

        foldLabels = foldLabels - 1
        if vectorizeLabels:
          foldLabels = labelsToVectors(foldLabels, 7)

        labelFolds.append(foldLabels)


    data = np.vstack(tuple(dataFolds))
    if vectorizeLabels:
      labels = np.vstack(tuple(labelFolds))
    else:
      labels = np.hstack(tuple(labelFolds))

  else:
    if big:
      fileName = 'equalized_kanade_big.pickle'
    else:
      fileName = 'equalized_kanade_small.pickle'

    # If there are no files with the equalized data, make one now
    if not os.path.exists(fileName):
      equalizeKanade(big)

    with open(fileName, "rb") as  f:
      data = pickle.load(f)
      labels = pickle.load(f)

    if not vectorizeLabels:
      labels = np.argmax(labels, axis=1)


  # For now: check that the data is binary
  assert np.all(np.min(data, axis=1) >= 0.0) and np.all(np.max(data, axis=1) < 1.0 + 1e-8)

  print "length of the Kande dataset"
  print len(data)

  # Fix the seed, so that we always get the same train and test set.
  # Before we fix the seed we need a truly random seed for when we want
  # to go back to random states.
  futureSeed = numpy.random.randint
  numpy.random.seed(0)
  data, labels = shuffle(data, labels)
  # Ensure the seed is not random after we exit this function
  numpy.random.seed(futureSeed)

  nrTrain = len(data) * 9 / 10
  if train:
    return np.array(data)[0:nrTrain], labels[0:nrTrain]
  else:
    return np.array(data)[nrTrain:], labels[nrTrain:]


def mapKanadeToPIELabels(kanadeData, kanadeLabels):
  kanadeToPie = {
    0: 2,
    1: 4,
    2: 5,
    3: 3,
    4: -1,
    5: 1,
    6: 0
  }

  print "kanadeLabels"
  print kanadeLabels

  # If we did not move to 0 to 6, do it now
  if 7 in kanadeLabels:
    kanadeLabels = kanadeLabels - 1

  mappedLabels = np.array(map(lambda x: kanadeToPie[x], kanadeLabels))
  # Keep the indices for emotions that do not map to the right
  # emotions in the PIE dataset
  keepIndices = mappedLabels != -1

  return kanadeData[keepIndices], mappedLabels[keepIndices]

# No labels: only used for unsupervised training
def readCroppedYale(equalize, globalPath=PATH):
  path = os.path.join(globalPath, "CroppedYale")

  imageFiles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(PATH)
    for f in fnmatch.filter(files, '*.pgm')]

  # Filter out the ones that containt "amyes bient"
  imageFiles = [ x for x in imageFiles if not "Ambient" in x]

  images = []
  for f in imageFiles:
    img = cv2.imread(f, 0)

    if equalize:
      img = equalizeImgGlobal(img)

    img = resize(img, SMALL_SIZE)

    images += [img.reshape(-1)]

  return np.array(images)

def readCroppedYaleSubjects(equalize=False, show=False, globalPath=PATH):
  path = os.path.join(globalPath, "CroppedYale")
  subject = 0
  subjectsToImgs = {}
  for subjectImagePath in os.listdir(PATH):
    print "subjectImagePath"
    print subjectImagePath


    fullSubjectPath = os.path.join(PATH, subjectImagePath)
    imageFiles = os.listdir(fullSubjectPath)
    imageFiles = [os.path.join(fullSubjectPath, f) for f in imageFiles]

    # Filter out the files which are not image files
    imageFiles = fnmatch.filter(imageFiles, '*.pgm')

    # Remove the ambient files
    imageFiles = [ x for x in imageFiles if not "Ambient" in x]

    print imageFiles

    images = []
    for f in imageFiles:
      img = cv2.imread(f, 0)
      print f
      print img.shape


      if equalize:
        img = equalizeImgGlobal(img)

      img = resize(img, SMALL_SIZE)

      if show:
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()

      images += [img.reshape(-1)]

    subjectsToImgs[subject] = images
    subject += 1

  return subjectsToImgs


# No labels: only used for unsupervised training
def readAttData(equalize=False, globalPath=PATH):
  path = os.path.join(globalPath, "att")

  imageFiles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(PATH)
    for f in fnmatch.filter(files, '*.pgm')]

  images = []
  for f in imageFiles:
    img = cv2.imread(f, 0)
    if equalize:
      img = equalizeImgGlobal(img)
    img = resize(img, SMALL_SIZE)
    images += [img.reshape(-1)]

  assert len(images) != 0

  return np.array(images)

# General methods for reading, cropping and equalizing input image data when
# we have no labels. The resulting images can only be used for unsupervised training.
def readCropEqualize(path, extension, crop, doRecognition, equalize=False,
                     isColoured=False):
  if not crop and doRecognition:
    raise Exception("you asked for the reading process to crop the images but do no face detection")

  if equalize:
    dirforres = "detection-cropped-equalized"
  else:
    dirforres = "detection-cropped"

  pathForCropped = os.path.join(path, dirforres)

  if crop:
    if doRecognition:
      if not os.path.exists(pathForCropped):
        os.makedirs(pathForCropped)

      imageFiles = [(os.path.join(dirpath, f), f)
        for dirpath, dirnames, files in os.walk(path)
        for f in fnmatch.filter(files, '*.' + extension)]

      images = []

      for fullPath, shortPath in imageFiles:
        # Do not do this for already cropped images
        if pathForCropped in fullPath:
          continue

        print fullPath
        print shortPath
        img = cv2.imread(fullPath, 0)
        print img == None

        face = facedetection.cropFace(img)


        if not face == None:

          face = resize(face, SMALL_SIZE)
          if equalize:
            face = equalizeFromFloatCLAHE(face)

          face = face.reshape(SMALL_SIZE)
          # Only do the resizing once you are done with the cropping of the faces
          # Check that you are always saving them in the right format
          print "face.min"
          print face.min()

          print "face.max"
          print face.max()

          assert face.min() >=0 and face.max() <=1
          images += [face.reshape(-1)]

          # Save faces as files
          croppedFileName = os.path.join(pathForCropped, shortPath)
          io.imsave(croppedFileName, face)
    # If not doing face detection live
    else:
      images = []
      imageFiles = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(pathForCropped)
        for f in fnmatch.filter(files, '*.' + extension)]

      for f in imageFiles:
        img = cv2.imread(f, 0)
        if type(img[0,0]) == np.uint8:
          print "rescaling unit"
          img = img / 255.0

        img = resize(img, SMALL_SIZE)
        images += [img.reshape(-1)]
  # If not doing recognition here, just reading from the initial faces
  else:
    images = []
    imageFiles = [os.path.join(dirpath, f)
      for dirpath, dirnames, files in os.walk(path) if dirnames not in ["detection-cropped-equalized", "detection-cropped"]
      for f in fnmatch.filter(files, '*.' + extension)]

    for i in imageFiles:
      assert not "detection-cropped" in imageFiles

    for f in imageFiles:
      img = cv2.imread(f, 0)
      if type(img[0,0]) == np.uint8:
        print "rescaling unit"
        img = img / 255.0

      img = resize(img, SMALL_SIZE)

      if equalize:
        img = equalizeFromFloatCLAHE(img)

      images += [img.reshape(-1)]

  assert len(images) != 0

  print len(images)
  return np.array(images)


# No labels: only used for unsupervised training
def readJaffe(crop, detectFaces, equalize, globalPath=PATH):
  path = os.path.join(globalPath, "jaffe")
  return readCropEqualize(path , "tiff", crop, detectFaces, equalize=equalize,
                          isColoured=False)

# No labels: only used for unsupervised training
def readNottingham(crop, detectFaces, equalize, globalPath=PATH):
  path = os.path.join(globalPath, "nottingham")
  return readCropEqualize(path, "png", crop, detectFaces, equalize=equalize,
                          isColoured=False)

# No labels: only used for unsupervised training
def readAberdeen(crop, detectFaces, equalize, globalPath=PATH):
  path = os.path.join(globalPath, "Aberdeen")
  return readCropEqualize(path, "jpg", crop, detectFaces, equalize=equalize,
                           isColoured=True)


def makeMultiDatabasePlot():
  jaffe = readJaffe(True, True, False)
  nottingham = readNottingham(True, True, False)
  aber = readAberdeen(True, True, False)
  yale = readCroppedYale(False)
  # att = readAttData()

  jaffe = jaffe[0]
  nottingham = nottingham[0]
  aber = aber[0]
  # att = att[0]

  final = np.hstack((jaffe.reshape(SMALL_SIZE),
                    nottingham.reshape(SMALL_SIZE),
                    aber.reshape(SMALL_SIZE)))
                    # att.reshape(SMALL_SIZE)))
  plt.imshow(final, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()


def readKaggleCompetitionSmallDataset(equalize, train=True):
  # Got data from here:
  # https://inclass.kaggle.com/c/facial-keypoints-detector/data
  data = []
  labels = []
  i = 0
  with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
      if i > 0:
        # print row
        emotion = int(row[0])
        instance = np.fromstring(row[1], dtype=int, sep=' ')
        if equalize:
          instance = equalizeFromFloatGlobal(instance, (48, 48))

        data += [instance]
        labels += [emotion]
        plt.imshow(instance.reshape((48, 48)), cmap=plt.cm.gray)
        plt.show()
      i += 1

  labels = labelsToVectors(labels, 7)
  data = np.array(data)

  # Before we fix the seed we need a truly random seed for when we want
  # to go back to random states.
  futureSeed = numpy.random.randint
  numpy.random.seed(0)
  data, labels = shuffle(data, labels)
  # Ensure the seed is not random after we exit this function
  numpy.random.seed(futureSeed)

  nrTrain = len(data) * 9 / 10

  if train:
    return np.array(data)[0:nrTrain], labels[0:nrTrain]
  else:
    return np.array(data)[nrTrain:], labels[nrTrain:]


def readKaggleCompetitionUnlabelled():
  data = []
  i = 0
  with open('test.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
      if i > 0:
        # print row
        instance = np.fromstring(row[0], dtype=int, sep=' ')
        data += [instance]
        # plt.imshow(instance.reshape((48, 48)), cmap=plt.cm.gray)
        # plt.show()
      i += 1

  data = np.array(data)
  return data

def makeKagglePlot():
  data, labels = readKaggleCompetition(False)
  data, labels = shuffle(data, labels)
  labels = np.argmax(labels, axis=1)
  print np.unique(labels)

  final = []
  for i in xrange(7):
    print i
    instances = data[labels == i]
    print instances[0].shape
    final += [instances[0].reshape((48, 48))]

  final = np.hstack(final)


  final2 = []
  for i in xrange(7):
    print i
    instances = data[labels == i]
    print instances[0].shape
    final2 += [instances[2].reshape((48, 48))]

  final2 = np.hstack(final2)



  supAndUnsup = np.vstack((final, final2))
  plt.imshow(supAndUnsup, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

def makeKagglePlotUnlabelled():
  data = readKaggleCompetitionUnlabelled()
  # data = shuffle(data, np.zeros(data.shape))
  np.random.shuffle(data)

  final = []
  for i in xrange(7):
    final += [data[i].reshape((48, 48))]

  final = np.hstack(final)
  plt.imshow(final, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

  return final


def readBigKaggle():
  trainingData = []
  trainingLabels = []

  publicTestData = []
  publicTestLabels = []

  privateTestData = []
  privateTestLabels = []

  i = 0
  with open('fer2013/fer2013.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
      if i > 0:
        # print row
        emotion = int(row[0])
        instance = np.fromstring(row[1], dtype=int, sep=' ')
        if row[2]  == "PublicTest":
          publicTestData+= [instance]
          publicTestLabels+= [emotion]
        elif row[2]  == "Training":
          trainingData+= [instance]
          trainingLabels+= [emotion]
        elif row[2]  == "PrivateTest":
          privateTestData+= [instance]
          privateTestLabels+= [emotion]
        else:
          raise Exception("bad label in bigKaggle reading")


      i += 1

  return trainingData, trainingLabels, publicTestData, publicTestLabels, privateTestData, privateTestLabels


def readBigKaggleTrain():
  trainingData, trainingLabels, _, _, _, _ = readBigKaggle()
  # for instance in trainingData:
  #   plt.imshow(instance.reshape((48, 48)), cmap=plt.cm.gray)
  #   plt.show()
  return trainingData, labelsToVectors(trainingLabels, 7)

def readBigKaggleTestPublic():
  _, _, publicTestData, publicTestLabels, _, _ = readBigKaggle()
  # for instance in publicTestData:
  #   plt.imshow(instance.reshape((48, 48)), cmap=plt.cm.gray)
  #   plt.show()
  return publicTestData, labelsToVectors(publicTestLabels, 7)


def readBigKaggleTestPrivate():
  _, _, _, _, privateTestData, privateTestLabels = readBigKaggle()
  return privateTestData, labelsToVectors(privateTestLabels, 7)


if __name__ == '__main__':
  # path = '/home/aela/uni/project/Multi-PIE_Aligned/A_MultiPIE.mat'
  # readMultiPIE(show=True)
  # readMultiPieDifferentPoses([2], show=True, equalize=True)
  # readCroppedYaleSubjects(show=True)
  # makeMultiPieImagesForReport()
  # makeEqualizePics()
  # makeCrossDbPlot()
  # facedetectionMultiPie()
  # makeMultiPosesPlot()
  # makeKanadeImag
  # readKaggleCompetitionUnlabelled()
  # readKaggleCompetition(False)
  # makeKagglePlot()
  # readBigKaggleTestPrivate()
  # makeMultiDatabasePlot()
  # makeMultiIlluminationsPlot()
  makeMultiPiePlot()
