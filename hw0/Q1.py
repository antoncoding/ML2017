import numpy as np
import sys

f1 = sys.argv[1]
f2 = sys.argv[2]

with open(f1,mode='r') as f1:
    dataA = f1.read()
    matrix_A = np.matrix(dataA).reshape(1,50)

with open(f2, mode='r') as f2:
    dataB = f2.read()
    matrix_B = np.matrix(dataB).reshape(50,10)

matrix_C = matrix_A * matrix_B
matrix_C.sort()
f = open('ans_one.txt','w')
for i in range(0,matrix_C.shape[1]):
    print(matrix_C.item(i), file=f)