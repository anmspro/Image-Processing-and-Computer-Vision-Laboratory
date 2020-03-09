import cv2
import math
import numpy as np
from scipy.linalg import toeplitz


img = cv2.imread("/home/anms/Desktop/1607058 (Image)/2nd Lab (25.02.2020)/Assignment/einstein_1.jpg", 0)
print(img)
#print("No val")

kernel=np.array(([1,2,1],[2,4,2],[1,2,1]))
f_size = kernel.shape[0]
a = f_size//2


m1 = img.shape[0]     
n1 = img.shape[1]     
m2 = kernel.shape[0]    
n2 = kernel.shape[1]    
out_row = m1 + m2 - 1   
out_col = n1 + n2 - 1   
print("Output size: (", out_row,out_col,")")


Zero_Padded_F = np.pad(kernel, ((out_row - m2, 0), (0, out_col - n2)), 'constant', constant_values=0)
print(Zero_Padded_F)

toeplitz_list = []

for i in range(Zero_Padded_F.shape[0] - 1, -1, -1):   
    col = Zero_Padded_F[i, :]                       
    row = np.r_[col[0], np.zeros(n1 - 1)]             
    toeplitz_mat = toeplitz(col,row)
    toeplitz_list.append(toeplitz_mat)


col = range(1, Zero_Padded_F.shape[0] + 1)
row = np.r_[col[0], np.zeros(m1 - 1, dtype=int)]
doubly_indices = toeplitz(col, row)


toeplitz_shape = toeplitz_list[0].shape             
hg = toeplitz_shape[0] * doubly_indices.shape[0]
wg = toeplitz_shape[1] * doubly_indices.shape[1]
doubly_blocked_shape = [hg, wg]
doubly_blocked = np.zeros(doubly_blocked_shape)


b_h, b_w = toeplitz_shape                      
for j in range(doubly_indices.shape[0]):
    for k in range(doubly_indices.shape[1]):
        start_j = j * b_h
        start_k = k * b_w
        end_j= start_j + b_h
        end_k = start_k + b_w
        doubly_blocked[start_j: end_j, start_k:end_k] = toeplitz_list[doubly_indices[j,k] - 1]


def InputMatrixToVectorConversion(in_mat):
    input_h, input_w = in_mat.shape
    output_vector = np.zeros(input_h * input_w, dtype=in_mat.dtype)
    in_mat = np.flipud(in_mat)
    for m,row in enumerate(in_mat):
        start = m * input_w
        end = start + input_w
        output_vector[start:end] = row   
        return output_vector  

Vectorized_in_img = InputMatrixToVectorConversion(in_img)
print('Vectorized Input: ', Vectorized_in_img)


result_vector = np.matmul(doubly_blocked, Vectorized_in_img)
print('Result: ', result_vector)


def OutputVectorToMatrix(res_vec, output_shape):
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype = res_vec.dtype)
    for n in range(output_h):
        st = n * output_w
        nd = st + output_w
        output[n, :] = res_vec[st:nd]
        output = np.flipud(output)
        return output

out_shape = [out_row, out_col]
FinalOutput = OutputVectorToMatrix(result_vector, out_shape)
max_ = FinalOutput.max()
min_ = FinalOutput.min()

FinalOutput = FinalOutput/(max_ - min_)
print('Result of implemented method: \n', FinalOutput)

cv2.imshow('Input', img)
cv2.imshow('Output', FinaOutput)


cv2.waitKey(0)
cv2.destroyAllWindows()