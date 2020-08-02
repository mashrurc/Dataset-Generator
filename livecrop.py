#Import library required for Capture face.
# Should you wish to use this code for
#education purpose in your assignment or dissertation
# please use the correct citation and give credit where required.


import cv2
from keras.models import load_model
model = load_model('faces.h5')

import numpy as np
from keras.preprocessing import image

import os, os.path
DIR = '/unknownfaces'

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0
i=194
# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#  Above line normalTest
#classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#Above line test with different calulation
#classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
#classifier = cv2.CascadeClassifier('lbpcascade_frontalface.xml')


while True:
    i+=1
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,0) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    def run(index):
        test_image = image.load_img( '/unknownfaces/face.jpg', target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        r=model.predict(test_image)

        q=["1","2","3","4"]
        return(q[r[0].tolist().index(max(r[0]))])

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #Save just the rectangle faces in SubRecFaces
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "unknownfaces/face"+str(i)+".jpg"
        cv2.imwrite(FaceFileName, sub_face)
        index=0
        name = run(index)
        cv2.putText(im,name,(x-w+170,y-h+130), font, 1.5, (255,255, 255), 2, cv2.LINE_AA)


    # Show the image
    cv2.imshow('Face Crop',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27: #The Esc key
        break
