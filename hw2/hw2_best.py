import numpy as np
import pandas as pd
import sys

train_file_X = sys.argv[3]
train_file_y = sys.argv[4]
test_file_X = sys.argv[5]
prediction_file = sys.argv[6]


def sigmoid (x): return 1.0/(1.0 + np.exp(-x))    # activation function
def sigmoid_(x): return x * (1.0 - x)             # derivative of sigmoid
def tanh_(x): return (1.0 -x**2)
def read_file_and_normalize(file_name):
    data = pd.read_csv(file_name)
    for column in data:
        mean = data[column].mean()
        std = data[column].std()
        if std!=0:
            data[column] = data[column].apply(lambda x: (x-mean)/std)
    data.insert(0,'Ones',1)
    return data

X = read_file_and_normalize(train_file_X)
y = pd.read_csv(train_file_y,header=None)
X = np.array(X)
y = np.array(y)

inputLayerSize, hiddenLayerSize, ouputLayerSize = 107, 4, 1
lambda_c, seed, lr = 1, 777, 0.001
epochs = 4000
np.random.seed(seed)
W0 = np.random.uniform(0.0,0.01,size=(inputLayerSize, hiddenLayerSize))
W1 = np.random.uniform(0.0,0.01,size=(hiddenLayerSize, ouputLayerSize))
for i in range(epochs):
    l0 = X
    l1 = np.tanh(np.dot(l0, W0))               
    l2 = sigmoid(np.dot(l1, W1))  
    l2_error = l2 - y  
    l2_delta = l2_error * sigmoid_(l2)
    l1_error = l2_delta.dot(W1.T)
    l1_delta = l1_error * tanh_(l1)
    W1 -= lr* (l1.T.dot(l2_delta) + lambda_c*W1)
    W0 -= lr* (l0.T.dot(l1_delta) + lambda_c*W0)

X_test = read_file_and_normalize(test_file_X)
X_test = np.array(X_test)

answer_list = []
H_test = np.tanh(np.dot(X_test, W0))               
Z_test = sigmoid(np.dot(H_test, W1))

for n in range(len(X_test)):
    if Z_test[n] >= 0.5:
        answer_list.append(1)
    else:
        answer_list.append(0)

output = pd.DataFrame(data=answer_list)
output.columns = ['label']
output = output.rename_axis('id')
output.reset_index(level=0,inplace=True)
output['id'] = output['id'].apply(int)+1
output.to_csv(prediction_file,index=False)