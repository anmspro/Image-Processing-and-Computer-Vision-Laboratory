import numpy as np
import cv2
import math
from scipy.linalg import toeplitz


def matrix_to_vector(input):
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row   
    return output_vector


def vector_to_matrix(input, output_shape):
    output_h, output_w = output_shape
    print(output_shape)
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    output=np.flipud(output)
    return output


flt = int(input("Enter filter size: "))
pad = flt//2

def normal(cimg):
    min=np.min(cimg)
    max=np.max(cimg)
    m=255/(max-min)
    h=len(cimg)
    w=len(cimg[h-1])
    for i in range(h):
        for j in  range(w):
            cimg[i][j]=m*(cimg[i][j]-max)+255
    cimg=np.array(cimg,dtype='uint8')
    return cimg


def convolution_as_maultiplication(I, F):
    I_row, I_col = I.shape 
    print("Input image size")
    print(I.shape)

    F_row, F_col = F.shape
    print("Filter image size")
    print(F.shape)

    output_row = I_row + F_row - 1
    output_col = I_col + F_col - 1
    print("output row and column")
    print(output_row)
    print(output_col)

    F_zero_pad = np.pad(F, ((output_row - F_row, 0),
                               (0, output_col - F_col)),
                            'constant', constant_values=0)

    toeplitz_list = []
    for i in range(F_zero_pad.shape[0]-1, -1, -1): 
        c = F_zero_pad[i, :] 
        r = np.r_[c[0], np.zeros(I_col-1)] 
        toeplitz_m = toeplitz(c,r) 
        toeplitz_list.append(toeplitz_m)

    c = range(1, F_zero_pad.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row-1, dtype=int)]
    doubly_indices = toeplitz(c, r)

    
    toeplitz_shape = toeplitz_list[0].shape
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    
    b_h, b_w = toeplitz_shape 
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    
    vectorized_I = matrix_to_vector(I)
    
    
    result_vector = np.matmul(doubly_blocked, vectorized_I)

    
    out_shape = [output_row, output_col]
    output = vector_to_matrix(result_vector, out_shape)
    
    return output

input_image =  cv2.imread("einstein_1.jpg", cv2.IMREAD_GRAYSCALE)

I = input_image

F=np.array(([1,2,1],[2,4,2],[1,2,1]))
#F=np.array(([0,0,0],[0,0,1],[0,0,0]))
#F=np.array(([1,0,-1],[1,0,-1],[1,0,-1]))
res = convolution_as_maultiplication(I,F)

cv2.imshow("input", input_image)
cv2.waitKey(1)
mx = res.max()
mn = res.min()
res = res/(mx-mn)

cv2.imshow("output",res)


cv2.waitKey(0)
cv2.destroyAllWindows()
