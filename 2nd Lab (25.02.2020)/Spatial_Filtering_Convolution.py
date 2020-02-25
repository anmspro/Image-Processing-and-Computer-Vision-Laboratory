
import numpy as np
import cv2
import math

img = cv2.imread('input.jpg', 0) #IMREAD_GRAYSCALE = 0
img1 = cv2.imread('input.jpg', 0)

cv2.imshow('Input Image', img)

w1 = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
w2 = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
w3 = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]

w1_sum = 1/16

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        a = img.item(i, j)
        g=0
        #for p in range(5):
        for m in range(3):
            for n in range(3):
               # if (i-m)>=0 and (i-m)<img.shape[0]-1 and (j-n)>=0 and (j-n)<img.shape[1]-1:
               l = img.item(i-m, j-n)
               #print(m, n)
               g = g + w2[m][n] * l 
        #img1.itemset((i, j), g*w1_sum)
        img1.itemset((i, j), g)
        
cv2.imshow('Output Image', img1)

cv2.waitKey(0)       
cv2.destroyAllWindows()
