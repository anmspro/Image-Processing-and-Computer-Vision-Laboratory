# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:58:10 2020

@author: HP-NPC
"""

import numpy as np
import cv2
import math

img = cv2.imread('input.jpg', 0) #IMREAD_GRAYSCALE = 0
img1 = cv2.imread('input.jpg', 0)
img2 = cv2.imread('input.jpg', 0)
img3 = cv2.imread('input.jpg', 0)
img4 = cv2.imread('input.jpg', 0)
img5 = cv2.imread('input.jpg', 0)
img6 = cv2.imread('input.jpg', 0)
img7 = cv2.imread('input.jpg', 0)
img8 = cv2.imread('input.jpg', 0)

cv2.imshow('Input Image', img)


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        a = img.item(i, j)
        
        img1.itemset((i, j), a&1)
        img2.itemset((i, j), a&2)
        img3.itemset((i, j), a&4)
        img4.itemset((i, j), a&8)
        img5.itemset((i, j), a&16)
        img6.itemset((i, j), a&32)
        img7.itemset((i, j), a&64)
        img8.itemset((i, j), a&128)
        
        
cv2.imshow('One Bit Slicing', img1)
cv2.imshow('Two Bit Slicing', img2)
cv2.imshow('Three Bit Slicing', img3)
cv2.imshow('Four Bit Slicing', img4)
cv2.imshow('Five Bit Slicing', img5)
cv2.imshow('Six Bit Slicing', img6)
cv2.imshow('Seven Bit Slicing', img7)
cv2.imshow('Eight Bit Slicing', img8)


#cv2.imwrite('One Bit Slicing.jpg', img1)
#cv2.imwrite('Two Bit Slicing.jpg', img2)
#cv2.imwrite('Three Bit Slicing.jpg', img3)
#cv2.imwrite('Four Bit Slicing.jpg', img4)
#cv2.imwrite('Five Bit Slicing.jpg', img5)
#cv2.imwrite('Six Bit Slicing.jpg', img6)
#cv2.imwrite('Seven Bit Slicing.jpg', img7)
#cv2.imwrite('Eight Bit Slicing.jpg', img8)


cv2.waitKey(0)       
cv2.destroyAllWindows()
