from keras.models import load_model, Model
import numpy as np
import pandas as pd
from vis.visualization import visualize_saliency
from vis.utils import utils
import cv2

import keras
from keras import backend as K
K.set_image_data_format('channels_first')

model_file = "whole_model.h5"
data_file = "train.csv"
saliency_img_file = "saliency_map.png"
original_img_file = "p4_original.png"
size = 48
indices = [77,62,105,118]

data = pd.read_csv(data_file, nrows = np.max(indices) + 1)

# for predict
x = np.array([r[1].split() for r in data.values], dtype=float).reshape((data.shape[0], 1, size, size))
imgs = x[indices]
imgs_p = imgs/255

model = load_model(model_file)
pred_class = model.predict_classes(imgs_p)

heatmaps = []
origins = []
for img, pc in zip(imgs, pred_class):
  img_seed = img.reshape(48,48,1)
  heatmap = visualize_saliency(model, len(model.layers) - 1, [pc], img_seed, alpha=0.6)
  heatmaps.append(heatmap)
  origins.append(img_seed)

cv2.imwrite(saliency_img_file, utils.stitch_images(heatmaps, cols=7))
cv2.imwrite(original_img_file, utils.stitch_images(origins, cols=7))
