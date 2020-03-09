import numpy as np
from scipy.linalg import toeplitz

I = np.array([[1, 2, 3], [4, 5, 6]])
F = np.array([[10, 20], [30, 40]])
print('I: ', I.shape)
print('F: ', F.shape)


I_row, I_column = I.shape 
F_row, F_column = F.shape

output_row = I_row + F_row - 1
output_column = I_column + F_column - 1
print('output dimension:', output_row_num, output_col_num)


F_zero_padded = np.pad(F, ((output_row - F_row, 0), (0, output_column - F_column)), 'constant', constant_values=0)
print('F_zero_padded: ', F_zero_padded)


toeplitz_list = []
for i in range(F_zero_padded.shape[0]-1, -1, -1):
    c = F_zero_padded[i, :]
    r = np.r_[c[0], np.zeros(I_column-1)]
    toeplitz_m = toeplitz(c,r)
    toeplitz_list.append(toeplitz_m)
    print('F '+ str(i)+'\n', toeplitz_m)


c = range(1, F_zero_padded.shape[0]+1)
r = np.r_[c[0], np.zeros(I_row-1, dtype=int)]
doubly_indices = toeplitz(c, r)
print('doubly indices \n', doubly_indices)


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

print('doubly_blocked: ', doubly_blocked)



def matrix_to_vector(input):
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row
        
    return output_vector

vectorized_I = matrix_to_vector(I)
print('vectorized_I: ', vectorized_I)

result_vector = np.matmul(doubly_blocked, vectorized_I)
print('result_vector: ', result_vector)



def vector_to_matrix(input, output_shape):
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    output=np.flipud(output)
    return output


out_shape = [output_row, output_column]
my_output = vector_to_matrix(result_vector, out_shape)

print('Result of implemented method: \n', my_output)



from scipy import signal

lib_output = signal.convolve2d(I, F, "full")
print('Result using signal processing library\n', lib_output)

assert(my_output.all() == lib_output.all())



