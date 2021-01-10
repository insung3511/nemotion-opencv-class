# Importing the dependencies
import cv2
import imutils
import numpy as np

pts1=[] 
pts2=[]
count=0

def draw_circle(event,x,y,flags,param):
    global pts1, count
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x,y), 2, (255, 0, 0), 3)
        pts1.append([x,y])
        
        if(count!=3):
            pts2.append([x,y])
        
        elif(count==3):
            pts2.insert(2,[x,y])
        
        print(pts1[count])
        count+=1
        
Orginimg = cv2.imread('textonwall.jpg')
img = imutils.resize(Orginimg, width=400)
cv2.namedWindow('image')

cv2.setMouseCallback('image',draw_circle)

while(True):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
