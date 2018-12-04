import numpy as np
import os
import re
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import keras
import scipy
import cv2
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import load_model

# extract, process, and save bottleneck features

# load image data
sketch_folder = r'data/sketches3/'
raw_features_path = r'data/raw_features3.npy'
features_path = r'data/features3.npy'

iv3_input = (139, 139, 3)

n_sketches = len(os.listdir(sketch_folder))
images = np.empty((n_sketches, iv3_input[0], iv3_input[1], iv3_input[2]))
raw_features = np.empty((n_sketches, 128, 128))
for file in os.listdir(sketch_folder):
    image = cv2.imread(sketch_folder + file)
    resized = cv2.resize(image.copy(), (iv3_input[0], iv3_input[1]))
    resized = preprocess_input(resized)
    sketch_number = int((file.split(".")[0])[6:])
    images[sketch_number] = resized.copy()
    raw_features[sketch_number] = image[:, :, 0]
    print("Processing Image #" + (file.split(".")[0])[6:])

images = np.array(images)
# save raw features
raw_features.dump(raw_features_path)

# load InceptionV3 iv3_model

iv3path = "inceptionv3.h5"
if iv3path not in os.listdir():
    model = InceptionV3(weights='imagenet', include_top=True, input_shape=iv3_input)
    model = keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    model.save(iv3path)
else:
    model = load_model(iv3path)
# extract bottleneck features
features = model.predict(images, batch_size=64, verbose=1)
features.dump(features_path)

# features = np.squeeze(features)
#
# features = iv3_model.predict(inception_input_train)
# # features = np.squeeze(features)
