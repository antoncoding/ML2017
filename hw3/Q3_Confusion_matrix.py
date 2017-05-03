import numpy as np
import csv
import sys

from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adagrad
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json, load_model

img_rows, img_cols = 48, 48
K.set_image_dim_ordering('th')

opt = Adam(lr=0.00025)

def train_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1,48,48), activation='relu',padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Dropout(0.2))
   
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
   
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.35))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dropout(0.4))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(7, activation='softmax'))
   
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

model = train_model()
model.load_weights("model_weight.h5")

# use validation set to compute confusion matrix
with open('train.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
    next(data_iter)
    data = [data for data in data_iter]
t_file = np.array(data, dtype = str)
train_data = [[pic[0], np.array(pic[1].split(' ')).reshape(1,48,48)] for pic in t_file ]

print('Read File done.')
X_train, y_train, X_test, y_test = [],[],[],[]
for i in range(len(train_data)):
    if i%100 ==1:
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
y_raw = y_test
y_test = np_utils.to_categorical(y_test)

predictions = model.predict(X_test)
pred= []
for i in range(len(predictions)):
    pred.append(np.argmax(predictions[i]))

print(confusion_matrix(y_raw, pred))