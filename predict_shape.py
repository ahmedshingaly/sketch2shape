import cv2
import numpy as np
import os
import torch
import keras
from keras.applications.inception_v3 import preprocess_input


def predict_shape(sketch, iv3_model, mapping_model, gan_model):
    sketch = np.tile(sketch, (1, 1, 3)).T
    sketch = preprocess_input(sketch)
    features = iv3_model.predict(sketch, batch_size=1, verbose=1)
    latent_vec = mapping_model.predict(features, batch_size=1)[0]
    latent_vec = torch.Tensor(latent_vec)
    latent_vec = latent_vec.view(1, -1, 1, 1, 1)
    fake = gan_model(latent_vec)
    np_fake = fake.detach().numpy()
    voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))
    return threshold(voxels)


def threshold(arr, val=0.5):
    small = arr < 0.5
    arr[small] = 0
    arr[~small] = 1
    return arr
