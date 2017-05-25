import numpy as np
from sklearn.svm import LinearSVR
from sklearn.neighbors import NearestNeighbors
import _pickle as cPickle
import sys

data_file_path = sys.argv[1]
output_file_path = sys.argv[2]

def get_eigenvalues(data_set):
    SAMPLE = 100
    NEIGHBOR = 150 
    random_idx = np.random.permutation(data_set.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(algorithm='ball_tree',n_neighbors=NEIGHBOR).fit(data_set)

    avg_eig_val = []
    for idx in random_idx:
        distance, points = knbrs.kneighbors(data_set[idx:idx+1]) 

        # get original datapoint of nearest neibors
        nbrs = data_set[points[0,1:]] # the first one will be itself

        U, eig_vals, V = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        eig_vals /= eig_vals.max()
        avg_eig_val.append(eig_vals)
    avg_eig_val = np.array(avg_eig_val).mean(axis=0)
    return avg_eig_val

## Fit data to SVR model
# npzfile = np.load('large_data.npz')
# X = npzfile['X']
# y = npzfile['y']
# print('Data loaded')


# svr = LinearSVR(C=1)
# print('Fitting SVR model')
# svr.fit(X, y)


# # save the model
# with open('my_model.pkl', 'wb') as fid:
#     cPickle.dump(svr, fid)    


# # load my model
with open('my_model.pkl', 'rb') as fid:
    svr_loaded = cPickle.load(fid)


testdata = np.load(data_file_path)
test_X = []

for i in range(200):
    data_set = testdata[str(i)]
    vs = get_eigenvalues(data_set)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr_loaded.predict(test_X)


with open(output_file_path, 'w') as out_file:
    print('SetId,LogDim', file=out_file)
    for i, d in enumerate(pred_y):
        if d <= 0:
    	    d = 1
        print('{},{}'.format(i,np.log(np.round(d))), file=out_file)

