import numpy as np
import csv

from keras.models import Sequential, load_model
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adagrad
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.layers.normalization import BatchNormalization

img_rows, img_cols = 48, 48

K.set_image_dim_ordering('th')

with open('train.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
    next(data_iter)
    data = [data for data in data_iter]
t_file = np.array(data, dtype = str)
train_data = [[pic[0], np.array(pic[1].split(' ')).reshape(1,48,48)] for pic in t_file ]

print('Read File done.')
X_train, y_train, X_test, y_test , uData = [],[],[],[],[]
for i in range(len(train_data)):
    if i%5 ==1:
        X_test.append(train_data[i][1])
        y_test.append(train_data[i][0])

    elif i%5 ==2:
        X_train.append(train_data[i][1])
        y_train.append(train_data[i][0])
    else:
        uData.append(train_data[i][1])
        

X_train = np.array(X_train, dtype=float)
y_train = np.array(y_train, dtype=int)
X_test = np.array(X_test, dtype=float)
y_test = np.array(y_test, dtype=int)
uData = np.array(uData, dtype=float)

X_train = X_train / 255
X_test = X_test / 255
uData = uData/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

pre_train_epochs = 200
batch_size = 256

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
    model.add(Dropout(0.45))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(num_classes, activation='softmax'))
   
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

    

# build the model
model = train_model()
print(model.summary())

datagen = ImageDataGenerator(
    featurewise_center=False, 
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
print("Using generating Data")
model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=batch_size,seed=7),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=pre_train_epochs,
                    validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

uData_y = model.predict(uData)

print("adding unlabeled data...")

X_train = np.concatenate((X_train, uData ))
y_train = np.concatenate((y_train, uData_y ))

model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=batch_size,seed=7),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=pre_train_epochs,
                    validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))


model.save('semi_model.h5')


# write ouput file
with open('test.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
    next(data_iter)
    data = [data for data in data_iter]
test_data = np.array(data, dtype = str)
test_data = [[np.array(pic[1].split(' ')).reshape(48,48)] for pic in test_data]

test_data = np.array(test_data, dtype=float)
test_data /= 255

predictions = model.predict(test_data)
csv = open("changed.csv","w")
csv.write("id,label\n")
for i in range(len(predictions)):
    csv.write(str(i) + "," + str(np.argmax(predictions[i])) + "\n")


