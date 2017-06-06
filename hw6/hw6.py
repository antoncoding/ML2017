import pickle
import numpy as np
import pandas as pd

import sys

data_dir = sys.argv[1]
output_path = sys.argv[2]


with open('model/nQ.p', 'rb') as handle:
    nQ = pickle.load(handle)

with open('model/nP.p','rb') as handle:
    nP = pickle.load(handle)

nR = np.dot(nP,nQ.T)

train_data =pd.read_csv( data_dir +'train.csv')

R_df = train_data.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)

colum_list = list(R_df.columns)

with open(output_path,'w') as outfile:
    print('TestDataID,Rating',file=outfile)
    test_file = pd.read_csv( data_dir + 'test.csv')
    for idx, row in test_file.iterrows():
        try:
            rating = nR[row['UserID']-1, colum_list.index(row['MovieID'])]
        except:
            rating = 3
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        print('{},{}'.format(idx+1, rating),file=outfile)
        