import numpy as np
import pandas as pd
import sys

np.random.seed(0)

import pickle

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


def fbeta_score(y_true, y_pred, beta=2):
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

def load_file(traindata, testdata): 
    train_text = []
    test_text = []
    tags = []
    
    f = open(traindata,'r')
    next(f)
    for row in f:
        data = row.split('"')
        tags.append(data[1])
        train_text.append(data[2])

    f = open(testdata,'r')
    next(f)
    for row in f:
        test_text.append(row)
    
    return train_text, tags, test_text


def RNN_model():
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],
                            weights=[embedding_matrix],input_length = x.shape[1], trainable=False))
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

train_text, tags, test_text = load_file('train_data.csv','test_data.csv')
all_text = test_text + train_text  

with open('tokenizer_x.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('tokenizer_y.pickle', 'rb') as handle:
    tag_tokenizer = pickle.load(handle)


# tokenizer = Tokenizer(num_words = 200000, split=' ', filters='1234567890!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
# tokenizer.fit_on_texts(all_text)


x = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen= 300)
test = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen= 300)

# tag_tokenizer = Tokenizer(filters = '')
# tag_tokenizer.fit_on_texts(tags)

y = np.asarray(tag_tokenizer.texts_to_matrix(tags))


# Store data (serialize)


# embeddings_index = {}
# glove_data = 'glove.42B.300d.txt'
# f = open(glove_data)
# for line in f:
#     values = line.split()
#     word = values[0]
#     value = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = value
# f.close()
# print('Loaded %s word vectors.' % len(embeddings_index))

# EMBEDDING_DIMENSION = 300
# word_index = tokenizer.word_index      

with open('embedding.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)


opt = RMSprop(lr=0.001, decay=1e-6, clipvalue=0.5)

X_train, y_train, X_validation, y_validation = train_validation_split(x,y,4500)

model = RNN_model()
earlystopping = EarlyStopping(monitor='val_fmeasure', patience = 50, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath='best.hdf5',verbose=1,save_best_only=True,save_weights_only=True,monitor='val_fmeasure',mode='max')
model.fit(X_train, y_train, epochs=300, batch_size=256, validation_data=(X_validation, y_validation), callbacks=[earlystopping,checkpoint])
model.load_weights('best.hdf5')

out = model.predict(X_validation)

threshold = np.arange(0.1,0.25,0.01,dtype=float)
acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append(f1_score(y_validation[:,i], y_pred, average='micro'))
    acc   = np.array(acc)
    index = np.where(acc==acc.max())
    accuracies.append(acc.max())
    best_threshold[i] = threshold[index[0][0]]
    acc = []
y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_validation.shape[1])] for i in range(len(y_validation))])


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

f = open('outputs.csv', 'w')
f.write('id,tags\n')
index = 0
for ans in output:
    f.write(str(index)+','+str(ans)+'\n')
    index += 1
f.close()
