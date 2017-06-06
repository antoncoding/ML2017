import numpy as np
import pandas as pd

from sklearn import dummy, metrics, cross_validation, ensemble

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, Adagrad
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras

# train_data = pd.read_csv('data/train.csv',dtype=int)
import sys

data_dir = sys.argv[1]
output_path = sys.argv[2]

test_data = pd.read_csv( data_dir+'test.csv',dtype=int)

# train_data.MovieID = train_data.MovieID.astype('category')
# train_data.UserID = train_data.UserID.astype('category')

test_data.MovieID = test_data.MovieID.astype('category')
test_data.UserID = test_data.UserID.astype('category')


MODEL_FILE = 'model/keras_1_weight.h5'

# Count the movies and users
n_movies = 3952
n_users = 6040

# Also, make vectors of all the movie ids and user ids. These are
# pandas categorical data, so they range from 1 to n_movies and 1 to n_users, respectively.

# movieid = np.array(train_data.MovieID.values)
# userid = np.array(train_data.UserID.values)

# y = np.zeros((train_data.shape[0], 5))
# y[np.arange(train_data.shape[0]), train_data.Rating - 1] = 1

# y=np.array(train_data.Rating)


movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 40, trainable=True)(movie_input))
movie_vec = keras.layers.Dropout(0.5)(movie_vec)

# Same thing for the users
user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 40, trainable=True)(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)

# Next, we join them all together and put them
# through a pretty standard deep learning architecture
input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')
nn = keras.layers.Dropout(0.4)(keras.layers.Dense(512, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.4)(keras.layers.Dense(256, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.4)(keras.layers.Dense(128, activation='relu')(nn))

result = keras.layers.Dense(1, activation='relu')(nn)

callbacks = [EarlyStopping('val_loss', patience=30), 
             ModelCheckpoint(MODEL_FILE, save_best_only=True)]
             


model = kmodels.Model([movie_input, user_input], result)


model.compile(Adam(lr=0.001), loss='mean_squared_error')


# final_layer = kmodels.Model([movie_input, user_input], nn)
# movie_vec = kmodels.Model(movie_input, movie_vec)


# a_movieid, b_movieid, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(movieid, userid, y,test_size=0.05)
# model.fit([a_movieid, a_userid], a_y, epochs=200, batch_size=512, validation_data=([b_movieid, b_userid], b_y), callbacks=callbacks)

model.load_weights(MODEL_FILE)


test_movieid = np.array(test_data.MovieID.values)
test_userid = np.array(test_data.UserID.values)

prediction = model.predict([test_movieid,test_userid])


with open(output_path, 'w') as outfile:
    print('TestDataID,Rating',file=outfile)
    for idx, pred in enumerate(prediction):
        # print('{},{}'.format(idx+1, np.argmax(pred)+1),file=outfile)
        rating = pred[0]
        if rating > 5:
            rating = 5
        elif rating < 1:
            rating = 1
        print('{},{}'.format(idx+1, rating),file=outfile)