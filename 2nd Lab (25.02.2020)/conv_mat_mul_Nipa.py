# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:36:13 2020

@author: Nipa Anjum
"""
import numpy as np
import cv2
import math
from scipy.linalg import toeplitz


def matrix_to_vector(input):
#    Converts the input matrix to a vector 
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    # flip the input matrix up-down because last row should go first
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row   
    return output_vector


def vector_to_matrix(input, output_shape):
#    Reshapes the output of the maxtrix multiplication to the shape "output_shape"
    output_h, output_w = output_shape
    print(output_shape)
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    # flip the output matrix up-down to get correct result
    output=np.flipud(output)
    return output

flt = int(input("Enter filter size: "))
pad = flt//2

def normal(cimg):
    min=np.min(cimg)
    max=np.max(cimg)
#    print(min)
#    print(max)
    m=255/(max-min)
    h=len(cimg)
    w=len(cimg[h-1])
    for i in range(h):
        for j in  range(w):
            cimg[i][j]=m*(cimg[i][j]-max)+255
    cimg=np.array(cimg,dtype='uint8')
    return cimg

def gaussian(sigma):
#    sigma = 1
    gf = np.zeros((2*pad+1,2*pad+1))
    c=2*math.pi*sigma*sigma
    for i in range(-pad,pad+1):
        for j in range(-pad,pad+1):
            xy=-(i*i+j*j)/(2*sigma*sigma)
            d=np.exp(xy)/c
            gf[i+pad][j+pad] = d
    asum = np.sum(gf)
    gf = gf/asum
    print(gf)
    return gf

def laplacian():
    sigma = 0.6
    lf = np.zeros((2*pad+1,2*pad+1))
    c = math.pi*np.power(sigma,4)
    z = 2*sigma*sigma
    for i in range(-pad,pad+1):
        for j in range(-pad,pad+1):
            xy = -(i*i+j*j)/z
            d = -(((1+xy)*np.exp(xy))/c)
            lf[i+pad][j+pad] = d
#    asum = np.sum(lf)
#    lf = lf/asum
#    lf[pad][pad]*=-1
    
#    gf1 = np.zeros((2*pad+1,2*pad+1))
#    gf2 = np.zeros((2*pad+1,2*pad+1))
#    gf1 = gaussian(1)
#    gf2 = gaussian(1.5)
#    lf = gf2 - gf1
#    return lf
    print(lf)
    return lf
def first_grad():
    sigma = 0.6
#    sigma_sqr = sigma * sigma
    s = math.pi * 2 * np.power(sigma,4)
    c = sigma*sigma
    df = np.zeros((2*pad+1,2*pad+1))
    for x in range(-pad, pad+1):
        for y in range(-pad, pad+1):
            temp = (x*x + y*y)/(2*c)
            df[x+pad][y+pad] = - (math.exp(-temp) * (x+y))/s
#    asum = np.sum(df)
#    df = df/asum
    print(df)
    return df
#def sobel():
#    s1=np.ones((2*pad+1,1))
#    s2=np.ones((1,2*pad+1))
#    s1[pad+1][0]=2
#    

def convolution_as_maultiplication(I, F):
    # number of columns and rows of the input 
    I_row, I_col = I.shape 
    print("Input image size")
    print(I.shape)

    # number of columns and rows of the filter
    F_row, F_col = F.shape
    print("Filter image size")
    print(F.shape)

    #  calculate the output dimensions
    output_row = I_row + F_row - 1
    output_col = I_col + F_col - 1
    print("output row and column")
    print(output_row)
    print(output_col)

    # zero pad the filter
    F_zero_pad = np.pad(F, ((output_row - F_row, 0),
                               (0, output_col - F_col)),
                            'constant', constant_values=0)

    # use each row of the zero-padded F to creat a toeplitz matrix. 
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(F_zero_pad.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = F_zero_pad[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col-1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)
#        print('F '+ str(i)+'\n', toeplitz_m)

        # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_pad.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row-1, dtype=int)]
    doubly_indices = toeplitz(c, r)

    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    # convert I to a vector
    vectorized_I = matrix_to_vector(I)
    
    # get result of the convolution by matrix mupltiplication
    result_vector = np.matmul(doubly_blocked, vectorized_I)

    # reshape the raw rsult to desired matrix form
    out_shape = [output_row, output_col]
    output = vector_to_matrix(result_vector, out_shape)
    
    return output

input_image =  cv2.imread("einstein2.jpg", cv2.IMREAD_GRAYSCALE)

I = input_image
##print("Enter 1 for gaussian filter\nEnter 2 for laplacian filter")
##type = int(input("Enter filter type: "))
##F = gaussian()
#F = laplacian()
##F = Gradient_filter()
##F1=np.array([[0,1,0],[1,-4,1],[0,1,0]])
##F = gaussian()
print("1. Gaussian\n 2. Laplacias\n 3. Sobel\n 4. 1st gradient ")
type = int(input("Enter Filter Type: "))    
if type==1:
    F=gaussian(1)
elif type==2:
    F=laplacian()
elif type==3:
    F=sobel()
elif type==4:
    F=first_grad()
res = convolution_as_maultiplication(I,F)
#res1 = convolution_as_maultiplication(I,F1)

cv2.imshow("input", input_image)
cv2.waitKey(1)
mx = res.max()
mn = res.min()
res = res/(mx-mn)
#res=normal(res)
#res1=normal(res1)
#print(res)
cv2.imshow("output",res)
#cv2.imshow("output1",res1)
cv2.waitKey(0)
cv2.destroyAllWindows()
