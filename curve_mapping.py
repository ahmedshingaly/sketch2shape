import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from utils.sketchify import show_image
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import cv2
import os
import torch
from test_funcs.util_vtk import visualization
from test_funcs.util import downsample


def imagify(contour, width=512):
    contour *= width
    print(contour)
    print(np.max(contour[0, :, 0]), np.max(contour[0, :, 1]))
    contour = contour.astype(int)
    return contour


if __name__ == "__main__":
    # data
    gan_features_path = r"data/samples_screenshot_BT.npy"
    features_path = r'data/features_vec.npy'

    # load GAN model
    basepath = r"models_cpu/"
    filepath = "chair_G_cpu"
    gan_model = torch.load(basepath + filepath)

    # other settings
    n_contours = 4
    OUTLINE_IMPORTANCE = 3

    X = np.load(features_path)
    X[:, 0, :, :] *= OUTLINE_IMPORTANCE
    print(X.shape)
    X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))  # reshape into a single column vector
    Y = np.load(gan_features_path)
    print(np.max(Y), np.min(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    print(X_train.shape)
    print('Fitting NN')
    NN = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', leaf_size=30, p=2, metric='minkowski')
    NN.fit(X_train)
    # NN.fit(X)
    # save model
    filename = 'nn_model'
    pickle.dump(NN, open(filename, 'wb'))
    distances, indices = NN.kneighbors(X_test)
    image_width = 512
    # exit()

    for crv, gan, index in zip(X_test, Y_test, indices):
        closest = index[0]
        print(closest)
        sketch = crv.reshape((n_contours, 200, 2))
        sketch[0, :, :] /= OUTLINE_IMPORTANCE
        sketch = imagify(sketch)
        im = cv2.drawContours(np.ones((np.max(sketch[0, :, 0]), np.max(sketch[0, :, 1]))) * 255,
                              np.array(sketch, dtype=np.int32),
                              -1, (0, 0, 0), 2)
        show_image(im)
        closest_sketch = X_train[closest, :].reshape((n_contours, 200, 2))
        closest_sketch[0, :, :] /= OUTLINE_IMPORTANCE
        closest_sketch = imagify(closest_sketch)
        im = cv2.drawContours(np.ones((np.max(closest_sketch[0, :, 0]), np.max(closest_sketch[0, :, 1]))) * 255,
                              np.array(closest_sketch, dtype=np.int32), -1, (0, 0, 0), 2)
        show_image(im)

        latent_vec = Y_train[closest]
        latent_vec = torch.Tensor(latent_vec)
        latent_vec = latent_vec.view(1, -1, 1, 1, 1)

        fake = gan_model(latent_vec)
        np_fake = fake.detach().numpy()
        voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))

        voxels = downsample(voxels, 2, method='mean')
        visualization(voxels, 0.3, title=None, uniform_size=0, use_colormap=False, angle=0.3)

    print(distances, indices)


