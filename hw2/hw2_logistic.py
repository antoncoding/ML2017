import numpy as np
import math
import pandas as pd
import sys

train_file_X = sys.argv[3]
train_file_y = sys.argv[4]
test_file_X = sys.argv[5]
prediction_file = sys.argv[6]

def sigmoid(X,theta):
    z = np.dot(X,theta)   
    return 1.0 / (1.0 + np.exp(-z))

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

train_size = X.shape[0]
feature_count = X.shape[1]

X = np.array(X)
y = np.array(y)

theta = np.zeros(feature_count)
sigma = np.ones(feature_count)
num_iters = 800

lr, r = 0.017, 0

for i in range(num_iters+1):
	for n in range(len(X)):
		h = sigmoid(X[n],theta)

		delta = y[n] - h       
		grad = (-1.0)*delta*(X[n])

		sigma = sigma + grad**2  
		theta = theta - (lr/np.sqrt(sigma))*grad
	
	if i % 50 == 0:
		s =0
		for n in range(len(X)):
			h = sigmoid(X[n],theta)
			if h>=0.5:
				if y[n] == 1:
					s += 1
			else:
				if y[n] ==0:
					s += 1
		print("Rate {}:\t {} %  ".format(i, 100*s/len(X)))	

X_test = read_file_and_normalize(test_file_X)
# X_test = X_test.drop('fnlwgt', axis=1)
X_test = np.array(X_test)

answer_list = []
for n in range(len(X_test)):
    if sigmoid(X_test[n], theta) >= 0.5:
        answer_list.append(1)
    else:
        answer_list.append(0)
print('Answer List created..')

output = pd.DataFrame(data=answer_list)
output.columns = ['label']
output = output.rename_axis('id')
output.reset_index(level=0,inplace=True)
output['id'] = output['id'].apply(int)+1
output.to_csv(prediction_file,index=False)
