import numpy as np
import csv
import sys

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

test_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# def train_model():
#     # create model
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), input_shape=(1,48,48), activation='relu',padding='same'))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
#     model.add(Dropout(0.2))
    
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(Dropout(0.3))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(Dropout(0.35))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(Dropout(0.3))
    
#     model.add(Flatten())
#     model.add(Dropout(0.4))
    
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.45))
#     model.add(Dense(num_classes, activation='softmax'))
   
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#     return model

    

# build the model
# model = train_model()


# datagen = ImageDataGenerator(
#     rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=False)  # randomly flip images

# datagen.fit(X_train)
# print("Using generating Data")
# model.fit_generator(datagen.flow(X_train, y_train,
#                                  batch_size=batch_size,seed=7),
#                     steps_per_epoch=X_train.shape[0] // batch_size,
#                     epochs=pre_train_epochs,
#                     validation_data=(X_test, y_test))

model = load_model('whole_model.h5')

# # write ouput file
with open(test_file_path,'r') as dest_f:
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
    next(data_iter)
    data = [data for data in data_iter]
test_data = np.array(data, dtype = str)
test_data = [[np.array(pic[1].split(' ')).reshape(48,48)] for pic in test_data]

test_data = np.array(test_data, dtype=float)
test_data /= 255

predictions = model.predict(test_data)
csv = open(output_file_path,"w")
csv.write("id,label\n")
for i in range(len(predictions)):
    csv.write(str(i) + "," + str(np.argmax(predictions[i])) + "\n")