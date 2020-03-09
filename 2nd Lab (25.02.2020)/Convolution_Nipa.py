import cv2
import numpy as np
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
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    output=np.flipud(output)
    return output




def gaussiankernel ():
    kernel = np.zeros((m, n), dtype=float)
    sigma = float(input("Enter sigma value: "))
    
    sigma = math.pow(sigma, 2)
    x = m//2
    y = n//2
    
    for a in range(m):
        for b in range(n):
            kernel[a][b] = (1/(2*3.1416*sigma))*(math.exp(-((math.pow(a-x, 2))+(math.pow(b-y, 2)))/(2*sigma)))
    summ = np.sum(kernel)
    for a in range(m):
        for b in range(n):
            kernel[a][b] = kernel[a][b]/summ
    return kernel


img = cv2.imread('lena.jpg',0)
cv2.imshow('input',img)

'''
n = int(input("Enter kernel size x: "))
m = int(input("Enter kernel size y: "))

kernel=gaussiankernel()

#print(k)

gaussianfilter = cv2.GaussianBlur(img,(n,m),0)

cv2.imshow('gaussian',gaussianfilter)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian1 = laplacian/laplacian.max()


outimg = cv2.filter2D(img,-1,k)
'''

kernel = np.array([[1,2,1],[2,4,2],[1,2,1]]) 

img_row_num, img_col_num = img.shape 
k_row_num, k_col_num = kernel.shape

output_row_num = img_row_num + k_row_num - 1
output_col_num = img_col_num + k_col_num - 1

k_zero_padded = np.pad(kernel, ((output_row_num - k_row_num, 0),(0, output_col_num - k_col_num)),
                       'constant', constant_values=0)

print('k_zero_padded: ', k_zero_padded)

toeplitz_list = []
for i in range(k_zero_padded.shape[0]-1, -1, -1): 
    col = k_zero_padded[i, :] 
    row = np.r_[col[0], np.zeros(img_col_num-1)] # first row for the toeplitz fuction should be defined 
    toeplitz_m = toeplitz(col,row) 
    toeplitz_list.append(toeplitz_m)
    #print('k '+ str(i)+'\n', toeplitz_m)



c = range(1, k_zero_padded.shape[0]+1)
r = np.r_[c[0], np.zeros(img_row_num-1, dtype=int)]
doubly_indices = toeplitz(c, r)
#print('doubly indices \n', doubly_indices)


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

#print('doubly_blocked: ', doubly_blocked)
        
        
vectorized_img = matrix_to_vector(img)
#print('vectorized_img: ', vectorized_img)

result_vector = np.matmul(doubly_blocked, vectorized_img)
#print('result_vector: ', result_vector)

out_shape = [output_row_num, output_col_num]
output = vector_to_matrix(result_vector, out_shape)
print(output)
#outputimg = int(np.divide(output,16))

for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        output[i][j]=round(output[i][j]/16)

#cv2.imshow('output',output)

cv2.waitKey(0)
cv2.destroyAllWindows()