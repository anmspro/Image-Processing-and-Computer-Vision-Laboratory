import cv2
import numpy
import math

n = int(input("Enter kernel size x: "))
m = int(input("Enter kernel size y: "))

kernel = numpy.zeros((m, n), dtype=float)

for a in range(m):
    for b in range(n):
        kernel[m-1-a][n-1-b] = float(input()) 

n2 = int(input("Enter kernel2 size x: "))
m2 = int(input("Enter kernel2 size y: "))

kernel2 = numpy.zeros((m, n), dtype=float)
kernel3 = numpy.zeros((m, n), dtype=float)
for a in range(m2):
    for b in range(n2):
        kernel2[a][b] = float(input())


n3= int(input("Enter kernel3 size x: "))
m3 = int(input("Enter kernel3 size y: "))

for a in range(m2):
    for b in range(n2):
        kernel3[m3-1-a][n3-1-b] = float(input())

print(kernel)
print("\n")
print(kernel2)
print("\n")
print(kernel3)

summ = numpy.sum(kernel)
print(summ)
kernel /= summ

img = cv2.imread('lenna.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", img)
img1=cv2.filter2D(img,-1,kernel)
#cv2.normalize(img1,0,255,cv2.NORM_MINMAX)
cv2.imshow('Output1', img1)


for a in range(m):
    for b in range(n):
        kernel[a][b] = kernel2[a][b]

ct=0

while ct<5:
    img = cv2.imread('lenna2.tif', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original", img)
    img2=cv2.filter2D(img,-1,kernel)
    cv2.imwrite('lenna2.tif', img2)
    ct=ct+1
    
cv2.imshow('Output2', img2)



for a in range(m):
    for b in range(n):
        kernel[a][b] = kernel3[a][b]

img = cv2.imread('lenna.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", img)
img3=cv2.filter2D(img,-1,kernel)
cv2.normalize(img3,0,255,cv2.NORM_MINMAX)
cv2.imshow('Output3', img3)


cv2.waitKey(0)
cv2.destroyAllWindows()