import cv2
import numpy as np
import os
import torch
import keras
from keras.applications.inception_v3 import preprocess_input
from test_funcs.util import downsample
from test_funcs.util_vtk import visualization
import pickle
from utils.sketchify import process_rgb_sketch, process_contours

OUTLINE_IMPORTANCE = 3  # if changed, must retrain model


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
    voxels = downsample(voxels, 4, method='max')
    return threshold(voxels)


def predict_nearest_shape(bgr_sketch, NNmodel, gan_model, latent_vectors, n_neighbors=5,
                          colors=[[255, 255, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255]]):
    # load data
    gan_features_path = r"data/samples_screenshot_BT.npy"
    # process rgb sketch
    contours = process_rgb_sketch(bgr_sketch, colors=colors,
                                  precision=0.00001, smoothing=10, resolution=200, diff_threshold=30, bspline_degree=1,
                                  n_contours=1)
    contours = process_contours(contours[:, :, 0, :])
    contours[0, :, :] *= OUTLINE_IMPORTANCE

    x = np.reshape(contours, (1, contours.shape[0] * contours.shape[1] * contours.shape[2]))

    # predict nearest neighbors

    closest = indices[0][0]
    # generate model

    latent_vec = latent_vectors[closest]
    latent_vec = torch.Tensor(latent_vec)
    latent_vec = latent_vec.view(1, -1, 1, 1, 1)

    fake = gan_model(latent_vec)
    np_fake = fake.detach().numpy()
    voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))
    voxels = downsample(voxels, 2, method='mean')
    return threshold(voxels, 0.4)


def threshold(arr, val=0.5):
    small = arr < 0.5
    arr[small] = 0
    arr[~small] = 1
    return arr


if __name__ == "__main__":
    OUTLINE_IMPORTANCE = 3  # if changed, must retrain model
    # load data
    gan_features_path = r"data/samples_screenshot_BT.npy"
    features_path = r'data/features_vec.npy'  # not req'd

    latent_vectors = np.load(gan_features_path)

    # load image
    image_path = r"utils/test_sketch.png"
    image = cv2.imread(image_path)

    # load nn model
    nn_model_path = "nn_model"  # 5 nn model
    nn_model = pickle.load(open(nn_model_path, 'rb'))

    # load GAN model
    basepath = r"models_cpu/"
    filepath = "chair_G_cpu"
    gan_model = torch.load(basepath + filepath)

    # process rgb sketch
    contours = process_rgb_sketch(image, colors=[[255, 255, 255], [0, 255, 48], [236, 0, 39], [0, 0, 255]],
                                  precision=0.00001, smoothing=10, resolution=200, diff_threshold=30, bspline_degree=1,
                                  n_contours=1)
    contours = process_contours(contours[:, :, 0, :])
    contours[0, :, :] *= OUTLINE_IMPORTANCE

    x = np.reshape(contours, (1, contours.shape[0] * contours.shape[1] * contours.shape[2]))

    # predict nearest neighbors
    nneighbors = 1
    distances, indices = nn_model.kneighbors(x)
    closest = indices[0][:nneighbors]
    distances = np.array([(1 / np.exp((x + 1) ** 2)) for x in range(nneighbors)]).reshape(1, -1)
    print(closest, distances[0])

    # generate model

    latent_vecs = latent_vectors[closest]
    latent_vec = np.sum(np.multiply(latent_vecs, (1 / distances).T), axis=0) / np.sum(1 / distances)
    latent_vec = torch.Tensor(latent_vec)
    latent_vec = latent_vec.view(1, -1, 1, 1, 1)

    fake = gan_model(latent_vec)
    np_fake = fake.detach().numpy()
    voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))

    voxels = downsample(voxels, 2, method='mean')
    visualization(voxels, 0.3, title=None, uniform_size=0, use_colormap=False, angle=0.3)
