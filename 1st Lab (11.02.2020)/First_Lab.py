import numpy as np
import cv2
import math

img = cv2.imread('input.jpg', 0) #IMREAD_GRAYSCALE = 0
img1 = cv2.imread('input.jpg', 0)
img2 = cv2.imread('input.jpg', 0)
img3 = cv2.imread('input.jpg', 0)

cv2.imshow('input image', img)
print(img.shape)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        a = img.item(i, j)
        #print(a)
        #img.itemset((i, j), 255-a) #Negative Transformation
        #img1.itemset((i, j), 31.87*math.log(a+1)) #Log Transformation
        #img2.itemset((i, j), math.pow(2, a/31.87)-1) #Inverse Log Transformation
        img3.itemset((i, j), 255*math.pow(a/255, 12)) #Gamma Transformation
        
#cv2.imshow('Output1', img1)
#cv2.imshow('Output2', img2)
cv2.imshow('Output3', img3)

#cv2.imwrite('Output1.jpg', img1)
#cv2.imwrite('Output2.jpg', img2)
#cv2.imwrite('Output3.jpg', img3)

cv2.waitKey(0)       
cv2.destroyAllWindows()
