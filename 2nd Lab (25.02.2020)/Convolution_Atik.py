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

img = cv2.imread('lenna.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", img)
height = img.shape[0]
width = img.shape[1]
image_arr = numpy.zeros((height+n-1, width+m-1), dtype=float)
image_arr[0:height, 0:width] = img

for i in range(height):
    for j in range(width):
        image_value = []
        k = 0
        z = 0
        for x in range(i, i+m):
            for y in range(j, j+n):
                image_value.append(image_arr[x][y]*kernel[k][z])
                z = z + 1
            k = k + 1
            z = 0
        s = sum(image_value)
        img[i][j] = float(s/summ)

cv2.imshow('Output1', img)



for a in range(m):
    for b in range(n):
        kernel[a][b] = kernel2[a][b]


summ = numpy.sum(kernel)
print(summ)


ct=0

while ct<5:
    img = cv2.imread('lenna2.tif', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original", img)
    height = img.shape[0]
    width = img.shape[1]
    image_arr = numpy.zeros((height+n-1, width+m-1), dtype=float)
    image_arr[0:height, 0:width] = img

    for i in range(height):
        for j in range(width):
            image_value = []
            k = 0
            z = 0
            for x in range(i, i+m):
                for y in range(j, j+n):
                    image_value.append(image_arr[x][y]*kernel[k][z])
                    z = z + 1
                k = k + 1
                z = 0
            s = sum(image_value)
            img[i][j] = float(s/summ)
    
    cv2.imwrite('lenna2.tif', img)
    ct=ct+1
    
cv2.imshow('Output2', img)




for a in range(m):
    for b in range(n):
        kernel[a][b] = kernel3[a][b]
img = cv2.imread('lenna.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", img)
height = img.shape[0]
width = img.shape[1]
image_arr = numpy.zeros((height+n-1, width+m-1), dtype=float)
image_arr[0:height, 0:width] = img

for i in range(height):
    for j in range(width):
        image_value = []
        k = 0
        z = 0
        for x in range(i, i+m):
            for y in range(j, j+n):
                image_value.append(image_arr[x][y]*kernel[k][z])
                z = z + 1
            k = k + 1
            z = 0
        s = sum(image_value)
        img[i][j] = float(s/summ)

cv2.normalize(img,0,255,cv2.NORM_MINMAX)
cv2.imshow('Output3', img)

summ = numpy.sum(kernel)
print(summ)


cv2.waitKey(0)
cv2.destroyAllWindows()