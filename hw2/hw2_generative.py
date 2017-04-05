import numpy as np
import math
import pandas as pd
from numpy.linalg import inv, pinv
import sys

train_file_X = sys.argv[3]
train_file_y = sys.argv[4]
test_file_X = sys.argv[5]
prediction_file = sys.argv[6]

np.set_printoptions(suppress=True)

def sigmoid(z):
    '''compute sigmoid value'''    
    return 1.0 / (1.0 + np.exp(-z))

X = pd.read_csv(train_file_X)
y = pd.read_csv(train_file_y,header=None)
X_test = pd.read_csv(test_file_X)

# Normalization X and X-test
for column in X:
    mean = X[column].mean()
    std = X[column].std()
    X[column] = X[column].apply(lambda x: (x-mean)/std)
for column in X_test:
    mean = X_test[column].mean()
    std = X_test[column].std()
    if std!= 0:
        X_test[column] = X_test[column].apply(lambda x: (x-mean)/std)
X = np.array(X)
y = np.array(y)
X_test = np.array(X_test)

# put data into class1 and class0
class1 = []
class0 = []
for n in range(len(y)):
    if y[n] == 1:
        class1.append(X[n])
    else:
        class0.append(X[n])

count_class_one = len(class1)
count_class_zero = len(class0)
class1 = np.array(class1)
class0 = np.array(class0)

mu0 = np.zeros(106)
mu1 = np.zeros(106)
mu0 = np.average(class0, axis=0)
mu1 = np.average(class1, axis=0)

# Compute sigma matrix of class1 and class0
sigma0 = np.zeros(shape=(106,106))
sigma1 = np.zeros(shape=(106,106))

for r in range(count_class_one):
    v = (class1[r]- mu1)
    sigma1 += v.reshape(106,1)*v.reshape(1,106)
for r in range(count_class_zero):
    v = (class0[r]- mu0)
    sigma0 += v.reshape(106,1)*v.reshape(1,106)

sigma0 /= count_class_zero
sigma1 /= count_class_one

# Compute total sigma matrix (weight sigma sum of sigma1 and sigma0)
total_sigma = (count_class_one/len(X))*sigma1 + (count_class_zero)/len(X)*sigma0 

# use the equation to compute w and b
w = ((mu0 - mu1).reshape(1,106)).dot(inv(total_sigma))
b = (- 1/2*(mu0.reshape(1,106).dot(inv(total_sigma))).dot(mu0.reshape(106,1)) + 1/2*(mu1.reshape(1,106).dot(inv(total_sigma))).dot(mu1.reshape(106,1)) + np.log(count_class_zero/count_class_one ))

answer_list = []
count =0
for n in range(len(X_test)):
    if sigmoid(w.dot(X_test[n]) + b )> 0.5:
        answer_list.append(0)
    else:
        answer_list.append(1)
        count+=1
        
# Ouput to generative[% of 1].csv
output = pd.DataFrame(data=answer_list)
output.columns = ['label']
output = output.rename_axis('id')
output.reset_index(level=0,inplace=True)
output['id'] = output['id'].apply(int)+1
output.to_csv(prediction_file),index=False)