import numpy as np
import csv

from keras.models import Sequential, load_model
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adagrad
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

K.set_image_dim_ordering('th')

with open('train.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
    next(data_iter)
    data = [data for data in data_iter]
t_file = np.array(data, dtype = str)
train_data = [[pic[0], np.array(pic[1].split(' ')).reshape(2304)] for pic in t_file ]

print('Read File done.')
X_train, y_train, X_test, y_test = [],[],[],[]
for i in range(len(train_data)):
    if i%5 ==1:
        X_test.append(train_data[i][1])
        y_test.append(train_data[i][0])
    else:
        X_train.append(train_data[i][1])
        y_train.append(train_data[i][0])

X_train = np.array(X_train, dtype=float)
y_train = np.array(y_train, dtype=int)
X_test = np.array(X_test, dtype=float)
y_test = np.array(y_test, dtype=int)

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

pre_train_epochs = 400
train_epochs = 200
batch_size = 512

def DNN_model():
    # create model
    model = Sequential()
    model.add(Dense(1024, input_dim=2304, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
   
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = DNN_model()
print(model.summary())


print('Using Train data.')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=train_epochs, batch_size=batch_size)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))


