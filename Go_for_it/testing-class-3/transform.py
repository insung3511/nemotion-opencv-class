import cv2
import imutils
import numpy as np

Originimg = cv2.imread('glass.jpg')
img = imutils.resize(Originimg, width=400)

pts1 = np.float32([[166, 138], [267, 103], [274, 381], [165, 383]])
pts2 = np.float32([[0, 0], [500, 0], [500, 1000], [0, 1000]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (500, 1000))

cv2.imshow("Original", img)
cv2.imshow("Transformed", dst)
cv2.waitKey(0)

cv2.destoryAllWindows()
