import cv2
import math
import numpy as np
from scipy.linalg import toeplitz

#step1: Define input and filter
in_img = cv2.imread("lenna.tif",cv2.IMREAD_GRAYSCALE)
print(in_img.shape)
kernel1=np.array(([1,2,1],[2,4,2],[1,2,1]))
f_size = kernel1.shape[0]
a = f_size//2


def convolution_as_matrix_multiplication(Filter):
    #step2: Calculate the final output size
    m1 = in_img.shape[0]     #height of the input image
    n1 = in_img.shape[1]     #width of the input image
    m2 = Filter.shape[0]    #height of the filter
    n2 = Filter.shape[1]    #width of the filter
    out_row = m1 + m2 - 1    #height of the output iamge
    out_col = n1 + n2 - 1    #width of the output image
    print("Output size: (", out_row,out_col,")")

    #step3: Zero-pad the filter matrix
    Zero_Padded_F = np.pad(Filter, ((out_row - m2, 0), (0, out_col - n2)), 'constant', constant_values=0)
    print(Zero_Padded_F)

    #Step4: Create Toeplitz matrix/Every row generates a toeplitz matrix
    toeplitz_list = []
    for i in range(Zero_Padded_F.shape[0] - 1, -1, -1):    #iterate from last row to the first row
        col = Zero_Padded_F[i, :]                          #ith row of the Zero_Padded_F 
        row = np.r_[col[0], np.zeros(n1 - 1)]              #first row for the toeplitz fuction should be defined otherwise the result is wrong
        toeplitz_mat = toeplitz(col,row)
        toeplitz_list.append(toeplitz_mat)
    #    print('F '+ str(i)+'\n', toeplitz_mat)

    #step5: Create a doubly blocked toeplitz matrix; which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    col = range(1, Zero_Padded_F.shape[0] + 1)
    row = np.r_[col[0], np.zeros(m1 - 1, dtype=int)]
    doubly_indices = toeplitz(col, row)
    #print('Doubly indices \n', doubly_indices)
    #create doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape                #shape of one toeplitz matrix
    hg = toeplitz_shape[0] * doubly_indices.shape[0]
    wg = toeplitz_shape[1] * doubly_indices.shape[1]
    doubly_blocked_shape = [hg, wg]
    doubly_blocked = np.zeros(doubly_blocked_shape)
    #tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape                              #height and widths of each block
    for j in range(doubly_indices.shape[0]):
        for k in range(doubly_indices.shape[1]):
            start_j = j * b_h
            start_k = k * b_w
            end_j= start_j + b_h
            end_k = start_k + b_w
            doubly_blocked[start_j: end_j, start_k:end_k] = toeplitz_list[doubly_indices[j,k] - 1]
   # print('doubly_blocked: ', doubly_blocked)

    #step6: Convert the input matrix to a column vector
    def InputMatrixToVectorConversion(in_mat):
        input_h, input_w = in_mat.shape
        output_vector = np.zeros(input_h * input_w, dtype=in_mat.dtype)
        in_mat = np.flipud(in_mat)                         #flip the input matrix up-down because last row should go first
        for m,row in enumerate(in_mat):
            start = m * input_w
            end = start + input_w
            output_vector[start:end] = row   
            return output_vector  
    Vectorized_in_img = InputMatrixToVectorConversion(in_img)
    print('Vectorized Input: ', Vectorized_in_img)

    #step7: Multiply doubly blocked toeplitz matrix with vectorized input
    result_vector = np.matmul(doubly_blocked, Vectorized_in_img)
    print('Result: ', result_vector)

    #step8: Reshape the result to a matrix form
    def OutputVectorToMatrix(res_vec, output_shape):
        output_h, output_w = output_shape
        output = np.zeros(output_shape, dtype = res_vec.dtype)
        for n in range(output_h):
            st = n * output_w
            nd = st + output_w
            output[n, :] = res_vec[st:nd]
            output = np.flipud(output)                          #flip the output matrix up-down to get correct result
            return output
    out_shape = [out_row, out_col]
    FinalOutput = OutputVectorToMatrix(result_vector, out_shape)
    max_ = FinalOutput.max()
    min_ = FinalOutput.min()
    FinalOutput = FinalOutput/(max_ - min_)
    print('Result of implemented method: \n', FinalOutput)
    return FinalOutput



out = convolution_as_matrix_multiplication(F)

cv2.imshow('Input',in_img)
cv2.imshow('Output', out)


cv2.waitKey(0)
cv2.destroyAllWindows()