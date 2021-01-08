import cv2
import imutils
import numpy as np

pts1 = []
pts2 = []
cnt  = 0

def draw_circle(event, x, y, flags, param):
    global pts1, cnt
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        pts1.append([x, y])

        if (cnt != 3):
            pts2.append([x, y])

        elif (cnt == 3):
            pts2.insert(2, [x, y])

        print(pts1[cnt])
        cnt += 1

Originimg = cv2.imread('glass.jpg')
img = imutils.resize(Originimg, width=400)
cv2.namedWindow('image')

cv2.setMouseCallback('image', draw_circle)

while (True):
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break


