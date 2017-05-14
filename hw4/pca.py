import numpy as np
import scipy
from PIL import Image
from matplotlib import pyplot as plt
from scipy.misc import imsave
# parameters
pixels = 64*64

def load_image(infilename) :
    img = Image.open(infilename )
    img.load()
    data = np.asarray(img, dtype="int32" )
    return data

def reconstruct(k, eig_pairs, save_file=False, plot_img=False):
    '''Reconstruct faces, using K eigen faces'''
    matrix_W = []
    for i in range(k):
        matrix_W.append(eig_pairs[i][1].reshape(4096))

    matrix_W = np.array(matrix_W).T
    transformed = matrix_W.T.dot(all_sample)
    
    reconstruct_imgs = []
    for i in range(100):
        img = matrix_W.dot(transformed[:,i])
        img += mean_vector
        img = img.reshape(64,64)
        reconstruct_imgs.append(img)
        if save_file:
            imsave('transformed/{}.png'.format(i),img)
    
    # plot_img = True: show image in console
    if plot_img:
        big_image = []
        plt.figure(figsize=(20,20))
        for i in range(100):
            big_image.append(plt.subplot(10,10,i+1))
            big_image[i].imshow(reconstruct_imgs[i].reshape(64,64),cmap='gray')
        plt.show()
    
    # RSME
    totalError = 0
    for i in range(100):
        error = matrix_W.dot(transformed[:,i]) + mean_vector - flatten_imgs[i]
        totalError += np.sum(error*error)

    totalError/= 409600
    totalError = np.sqrt(totalError)/2.56
    
    return totalError

def generate_eigen_faces(n_faces):
    eigen_faces = []
    # plot n faces
    for i in range(n_faces):
        img =eig_pairs[i][1].reshape(64,64)
        eigen_faces.append(img)
        imsave('eigen_faces/{}.png'.format(i),img)   

imgs =[]
# store imges in 2304 dimensions 
flatten_imgs = []
for cha in 'ABCDEFGHIJ':
    for indx in range(10):
        img = load_image('facedata/{}0{}.BMP'.format(cha,indx))
        imsave('origin/{}{}.png'.format(cha,indx),img)
        imgs.append(img)
        flatten_imgs.append(img.reshape(pixels))

all_sample = np.ndarray(shape=(4096,100))
for pixel in range(pixels):
    for indx in range(100):
        all_sample[pixel,indx]=flatten_imgs[indx][pixel]

list_to_cov=[] # to put in np.cov
mean_vector= np.mean(all_sample,axis=1)

for i in range(4096):
    for j in range(100):
        all_sample[i,j] -= mean_vector[i]

cov_mat = np.cov(all_sample)

eig_val, eig_vec = np.linalg.eig(cov_mat)

idx = eig_val.argsort()[::-1]
eig_vec = eig_vec.astype(float)

eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

generate_eigen_faces(n_faces=10)

reconstruct(59, eig_pairs, save_file=False,plot_img=False)