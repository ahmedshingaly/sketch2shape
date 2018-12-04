import numpy as np
import os
import re
import matplotlib.pyplot as plt
import keras
import scipy
import cv2
# from scipy.ndimage.interpolation import shift
from utils.sketchify import vectorify, normalize, shift


def normalize(contour, column=0):
    normalizer = np.max(contour[0, :, column]) - np.min(contour[0, :, column])
    contour[:, :, 0] = contour[:, :, 0] - np.min(contour[0, :, 0])
    contour[:, :, 1] = contour[:, :, 1] - np.min(contour[0, :, 1])
    return contour / normalizer


def shift(arr, val):
    if val != 0:
        new_arr = np.zeros(arr.shape)
        new_arr[val:, :] = arr[:-val, :]
        new_arr[:val, :] = arr[-val:, :]
        return new_arr
    else:
        return arr


if __name__ == "__main__":
    # load image data
    screenshot_folder = r'data/screenshotsBT/'
    gan_features_path = r"data/samples_screenshot_BT.npy"
    features_path = r'data/features_vec.npy'

    latent_vecs = np.load(gan_features_path)

    n_samples = latent_vecs.shape[0]
    n_features = 200
    n_contours = 4
    # 0: outline, black
    # 1: seat, green
    # 2: back, blue
    # 3: side, red

    raw_features = np.empty((n_samples, n_contours, n_features, 2))

    # for file in os.listdir(screenshot_folder):
    for i in range(10000):
        print(i)
        filename = screenshot_folder + "screenshot" + str(i) + ".png"
        # filename = screenshot_folder + file
        # image = cv2.imread(screenshot_folder + file)
        out_contour = vectorify(filename, precision=0.00001, smoothing=10, resolution=n_features,
                                colors=[255, 7, 42, 77], diff_threshold=5, bspline_degree=1,
                                n_contours=1)
        print(np.array(out_contour).shape)
        out_contour = out_contour[:, :, 0, :]
        for j in range(n_contours):
            min_y_ind = np.argmin(out_contour[j, :, 1])
            print(min_y_ind)
            out_contour[j] = shift(out_contour[j], min_y_ind)
        out_contour = normalize(out_contour)
        print(np.min(out_contour[:, :, 0]), np.max(out_contour[:, :, 0]))
        print(np.min(out_contour[:, :, 1]), np.max(out_contour[:, :, 1]))
        raw_features[i] = out_contour

    raw_features.dump(features_path)
