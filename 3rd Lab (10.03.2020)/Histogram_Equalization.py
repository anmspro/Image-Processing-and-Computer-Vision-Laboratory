import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
img1 = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
plt.hist(img1.ravel(), 256, [0, 256])
plt.show()

n = plt.hist(img1.ravel(), 256, [0, 256])
histo = np.histogram(img1.flatten(), 10, [0, 256])

print(histo[0])

img21= cv2.equalizeHist(img1)
cv2.imshow('Equalized Image', img21)

print(" Equalized CDF:\n")
plt.hist(img21.ravel(), 256, [0, 256])
plt.show()

ih = plt.hist(img.ravel(), 256, [0, 256])[0]

plt.hist(ih, 256, [0, 256])
'''

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Input Image', img)

print("Input Histogram:")
plt.hist(img.ravel(),256,[0,256])
plt.show()

#print('Input Image')

hk = np.zeros((256),dtype=float)
p = np.zeros((256),dtype=float)
s = np.zeros((256),dtype=int)
s2 = np.zeros((256),dtype=int)
t_pix = img.shape[0]*img.shape[1]
Px = 0

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hk[img[i][j]]+=1

for i in range (0,256):
    p[i]=hk[i]/(t_pix)
    Px=Px+p[i]
    #s2[i]=Px
    s[i]=round(255*Px)

#print(":\n")
#print(s)
fig = plt.figure()
fig.suptitle('Input CDF')
plt.plot(s)

plt.hist(s.ravel(), 256, [0,256])
plt.show()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        val = img[i][j]
        img.itemset((i,j),s[val])

print("Equalized Histogram:")
plt.hist(img.ravel(), 256, [0,256])
plt.show()

cv2.imshow('Equalized Image', img)

hk1 = np.zeros((256),dtype=float)
p1 = np.zeros((256),dtype=float)
s1 = np.zeros((256),dtype=int)
t_pix1 = img.shape[0]*img.shape[1]
Px1 = 0

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        hk1[img[i][j]]+=1

for i in range (0,256):
    p1[i]=hk1[i]/(t_pix1)
    Px1=Px1+p1[i]
    s1[i]=round(255*Px1)
    #s1[i]=round(Px1)

#cv2.normalize(s,0,255,cv2.NORM_MINMAX)
print("Equalized CDF:\n")
#print(s1)
plt.plot(s1)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()