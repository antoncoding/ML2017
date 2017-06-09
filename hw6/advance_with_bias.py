import numpy as np
import pandas as pd

COUNT_USER = 6040
COUNT_MOVIE = 3952

def matrix_factorization(R, P, Q, K,valid_point,steps=200, alpha=0.0005, beta=0.0002):
    Q = Q.T
    P_bias = np.zeros(COUNT_USER)
    Q_bias = np.zeros(COUNT_MOVIE)
    for step in range(steps):
        print(step)
        for [i, j] in valid_point:
            eij = R[i][j] - np.dot(P[i,:],Q[:,j]) - P_bias[i] - Q_bias[j]
            for k in range(K):
                P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
  
                P_bias[i] = P_bias[i] + alpha * 2 * eij
                Q_bias[j] = Q_bias[j] + alpha * 2 * eij
        
        eR = np.dot(P,Q)
        if step%10 ==0 :
            e = 0
            for [i, j] in valid_point:
                e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]) -P_bias[i] - Q_bias[j], 2)
                for k in range(K):
                    e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
            print('step {}: error: {}'.format(step,e))
            
    return P, Q.T, P_bias, Q_bias

train_data = pd.read_csv('data/train.csv',dtype=int)
test_data = pd.read_csv('data/test.csv',dtype=int)

# R_df = train_data.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
# R = R_df.as_matrix()

R2 = np.zeros(shape=(COUNT_USER,COUNT_MOVIE))

valid_point = []
for idx, row in train_data.iterrows():
    userid = row['UserID']
    movieid = row['MovieID']
    r = row['Rating']
    R2[userid-1, movieid-1] = r
    valid_point.append([userid-1,movieid-1])

N = len(R2)
M = len(R2[0])
K = 7

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ, P_bias, Q_bias = matrix_factorization(R2, P, Q, K, valid_point= valid_point)

import pickle

pickle.dump( nP, open( "nP_b.p", "wb" ))
pickle.dump( nQ, open( "nQ_b.p", "wb" ))
pickle.dump( P_bias, open( "P_bias.p", "wb" ))
pickle.dump( Q_bias, open( "Q_bias.p", "wb" ))