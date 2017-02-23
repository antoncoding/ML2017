import numpy as np
import sys

f1 = sys.argv[1]
f2 = sys.argv[2]

matrix_A = np.loadtxt(f1,dtype='int',delimiter=',')
matrix_B = np.loadtxt(f2,dtype='int',delimiter=',')
matric_C = np.dot(matrix_A, matrix_B)
output_matrix = np.sort(matric_C,axis=None)
f = open('ans_one.txt','w')
for item in output_matrix:
    print(item, file=f)
