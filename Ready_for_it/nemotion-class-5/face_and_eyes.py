import numpy as np
import cv2

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('image/isthatright?.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (startX, startY, endX, endY) in faces:
    cv2.rectangle(img, (startX, startY), (startX + endX, startY + endY), (255, 0, 0), 2)
    roi_gray = gray[startY : startY + endY, startX : startX + endX]
    roi_color = img[startY : startY + endY, startX : startX + endX]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (eyeX, eyeY, eyeW, eyeH) in eyes:
        cv2.rectangle(roi_color, (eyeX, eyeY), (eyeX + eyeW, eyeY + eyeH), (0, 255, 0), 2)

cv2.imshow('Result', img)

cv2.waitKey(0)
cv2.destroyAllWindows()