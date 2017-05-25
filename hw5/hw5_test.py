import numpy as np
import pandas as pd

np.random.seed(0)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.metrics import hamming_loss
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
import pickle
from keras import backend as K


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def RNN_model():
    model = Sequential()
    model.add(Embedding(45899,300,input_length = 300, trainable=False))
    model.add(GRU(512, recurrent_dropout = 0.5, dropout=0.5, return_sequences=True, activation='relu', implementation=2))
    model.add(GRU(256, recurrent_dropout = 0.5, dropout=0.5, return_sequences=True, activation='relu', implementation=2))
    model.add(GRU(128, recurrent_dropout = 0.5, dropout=0.5, activation='relu', implementation=2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(39, activation='sigmoid'))
    model.add(Dropout(0.3))
    model.compile(loss='binary_crossentropy', optimizer=opt,metrics=[fmeasure])
    
    return model

def train_validation_split(x,y,num_train):
    X_train = x[:num_train]
    y_train = y[:num_train]
    X_validation = x[num_train:]
    y_validation = y[num_train:]
    
    return X_train, y_train, X_validation, y_validation

testFile = sys.argv[1]
outFile = sys.argv[2]

test_text = []
f = open(testFile,'r')
next(f)
for row in f:
    test_text.append(row)

with open('tokenizer_x.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

test = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen= 300)

with open('tokenizer_y.pickle', 'rb') as handle:
    tag_tokenizer = pickle.load(handle)



opt = RMSprop(lr=0.001, decay=1e-6, clipvalue=0.5)

model = RNN_model()
model.load_weights('best.hdf5')

best_threshold = []

with open('threshold.pickle', 'rb') as handle:
    best_threshold = pickle.load(handle)


pred = model.predict(test)
result = np.array([[1 if pred[i,j]>=best_threshold[j] else 0 for j in range(pred.shape[1])] for i in range(len(pred))])

dic = tag_tokenizer.word_index 
label_list = [x.upper() for x in sorted(dic, key=dic.get)]

output = []
for temp in result:
    label = []
    for i in range(len(temp)):
        if temp[i] == 1:
            label.append(label_list[i-1])
    output.append(' '.join(label))


output = np.asarray(output)

f = open(outFile, 'w')
f.write('id,tags\n')
index = 0
for ans in output:
    f.write(str(index)+','+str(ans)+'\n')
    index += 1
f.close()
