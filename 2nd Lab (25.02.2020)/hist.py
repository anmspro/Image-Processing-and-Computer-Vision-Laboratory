# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:19:18 2020

@author: NLP Lab
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import copy

img = cv2.imread('einstein.jpg', cv2.IMREAD_GRAYSCALE)
img2 = np.zeros((img.shape[0],img.shape[1]))
img1=copy.copy(img)
plt.hist(img.ravel(),256,[0,256])
plt.show()
print('Input')

hk = np.zeros((256),dtype=float)
p = np.zeros((256),dtype=float)
s = np.zeros((256),dtype=int)
t_pix = img.shape[0]*img.shape[1]
Hx = 0

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hk[img[i][j]]+=1

for i in range (0,255):
    p[i]=hk[i]/(t_pix)
    Hx=Hx+p[i]
    s[i]=round(255*Hx)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        val = img[i][j]
        img.itemset((i,j),s[val])
#end
        
##to see cdf input
plt.hist(s.ravel(),256,[0,256])
plt.show()
print("cdf 1")
        
hk1 = np.zeros((256),dtype=float)
p1 = np.zeros((256),dtype=float)
s1 = np.zeros((256),dtype=int)
t_pix1 = img.shape[0]*img.shape[1]
Hx1 = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hk1[img[i][j]]+=1

for i in range (0,255):
    p1[i]=hk1[i]/(t_pix1)
    Hx1=Hx1+p1[i]
    s1[i]=round(255*Hx1)
#to see cdf input
plt.hist(s1.ravel(),256,[0,256])
plt.show()
print("cdf2")

plt.hist(img.ravel(),256,[0,256])
plt.show()
print('Output')
cv2.imshow('in', img1)
cv2.imshow('out',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

