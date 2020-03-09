import cv2
import numpy as np
import math


img = cv2.imread('ross.jpg', cv2.IMREAD_GRAYSCALE)
out= img.copy()
out2= img.copy()

m = int(input())
n = int(input())

kernel = []

for i in range(m):
    for j in range(n):
        temp = int(input())
        kernel.append(temp)
        
    
    
sum = 0
print(kernel)

s = [-1,0,1,-1,0,1,-1,0,1]
t = [-1,-1,-1,0,0,0,1,1,1]



for i in kernel:
    sum = sum+i
print(sum)



for i in range(img.shape[0]-1):
    for j in range(img.shape[1]-1):
        if i!=0 or j!=0:
            tot = 0
            for p in range(m*n):
                tot=tot+img[i-s[p]][j-t[p]]*kernel[p]
        
            a= img.item(i,j)   
            out.itemset((i,j),tot/sum)
            
                
                
        

#2

kernel.clear()

for i in range(m):
    for j in range(n):
        temp = int(input())
        kernel.append(temp)


sum = 0
print(kernel)

  
for i in kernel:
    sum = sum+i
print(sum)

out1=img.copy()

for ti in range(5):
    for i in range(out1.shape[0]-1):
        for j in range(out1.shape[1]-1):
            if i!=0 or j!=0:
                tot = 0
                for p in range(m*n):
                    tot=tot+out1[i+s[p]][j+t[p]]*kernel[p]
        
                a= out1.item(i,j)   
                out1.itemset((i,j),tot/sum)




#3
                
kernel.clear()

for i in range(m):
    for j in range(n):
        temp = int(input())
        kernel.append(temp)


sum = 0
print(kernel)



for i in range(img.shape[0]-1):
    for j in range(img.shape[1]-1):
        if i!=0 or j!=0:
            tot = 0
            for p in range(m*n):
                tot=tot+img[i-s[p]][j-t[p]]*kernel[p]
        
            a= img.item(i,j)   
            out2.itemset((i,j),tot)
            
cv2.normalize(out2,0,255,cv2.NORM_MINMAX)



cv2.imshow("Input image", img)
cv2.imshow("Output image1", out)
cv2.imshow("Output image2", out1)
cv2.imshow("Output image3", out2)
cv2.waitKey(0)
cv2.destroyAllWindows()