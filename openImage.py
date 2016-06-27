import numpy as np
import cv2
import glob 
from PIL import Image
from matplotlib import pyplot as plt
import os

##========================================================================================
##load image from a certain folder 
##path images : /home/gandalf/NudityFiltering/opencvPy/image/1.jpg
##load as numpy array
##========================================================================================
# pil_im = Image.open("/home/gandalf/NudityFiltering/opencvPy/image/1.jpg")
# print pil_im.format, pil_im.size, pil_im.mode
# np_im=np.array(pil_im)
# plt.imshow(np_im)
# plt.show()
##========================================================================================

##========================================================================================
##load multiple image in numpy image
##========================================================================================
# imageformat=".jpg"
# path="/home/gandalf/NudityFiltering/opencvPy/image/"
# imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
# for el in imfilelist:
#         print el
#         image = Image.open(el)
#         image = np.array(image)
#         plt.imshow(image)
#         plt.show()
##========================================================================================

##========================================================================================
##load multiple image in numpy image
##========================================================================================
imageformat=".jpg"
path="/home/gandalf/NudityFiltering/opencvPy/image/"
imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
for el in imfilelist:
		image = cv2.imread(el, cv2.CV_LOAD_IMAGE_COLOR)
		if (!image): 
				print "Image is not found"
				break
		cv2.imshow('Image', image) 
		cv2.waitKey(1000)
##========================================================================================



##========================================================================================
##load video in numpy image
##========================================================================================
# #load video file
# cap = cv2.VideoCapture(0)

# #open video
# while(True)
# 	#capture frame-by-frame
# 	ret, frame = cap.read()

# 	#convert image to grayscale
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 	#Display the resulting frame
# 	cv2.imshow('frame', gray)
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break
# cap.release()



##========================================================================================
##Haar Cascade Classifier
##========================================================================================
# #load image 
# image = cv2.imread("bukber.jpg")
# #initiate the cascade classifier
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# #convert image to grayscale
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# #build the cascade classifier
# face = face_cascade.detectMultiScale(gray,1.3,5)
# for (x,y,w,h) in face: 
# 		cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),1)
# 		roi_gray = gray[y:y+h, x:x+w]
# 		roi_color = image[y:y+h, x:x+w]
# 		eyes = eye_cascade.detectMultiScale(roi_gray)
# 		for (ex,ey,ew,eh) in eyes: 
# 			cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(255,0,255),1)
# #display an image
# cv2.imshow("image",image)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
