import cv2
import imutils
import numpy as np

imgOriginal = cv2.imread('textonwall.jpg')
img = imutils.resize(imgOriginal, width=400)

pts1 = np.float32([[56,184],[351, 176],[355,312],[56, 321]])
pts2 = np.float32([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (1000,500))

cv2.imshow("Original", img)
cv2.imshow("Transform", dst)
cv2.waitKey(0)

cv2.destroyAllWindows()