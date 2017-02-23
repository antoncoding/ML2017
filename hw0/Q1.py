import numpy as np
import sys
#
f1 = sys.argv[1]
f2 = sys.argv[2]

num_lines_A = sum(1 for line in open(f1))
num_lines_B = sum(1 for line in open(f2))

with open(f1) as f1:
    dataA = f1.read()
    num_cols_A = len(dataA)//(num_lines_A *2)

with open(f2, mode='r') as f2:
    dataB = f2.read()
    num_cols_B = len(dataB)//(num_lines_B *2)

matrix_A = np.matrix(dataA).reshape(num_lines_A, num_cols_A)
matrix_B = np.matrix(dataB).reshape(num_lines_B, num_cols_B)
matrix_C = matrix_A * matrix_B
print_list = []
f = open('ans_one.txt','w')
for j in range(0,num_cols_B):
    for i in range(0,num_lines_A):
        print_list.append(matrix_C[i,j])

print_list.sort()
for item in print_list:
    print(item, file=f)