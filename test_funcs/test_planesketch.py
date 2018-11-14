import torch as torch
from torch.legacy import nn as nn
import test_funcs.convertLegacy as c
import numpy as np
from test_funcs.util_vtk import visualization
from test_funcs.util import downsample
import vtk
import cv2
import matplotlib.pyplot as plt

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # we're dealing with cpu iv3_model

# load iv3_model
basepath = r"../models_cpu/"
filepath = "chair_G_cpu"

model = torch.load(basepath + filepath)

# generate random latent vectors
batch_size = 1
latent_size = 200

z = torch.randn(batch_size, latent_size, device=device)
z = z.view(1, -1, 1, 1, 1)
print(z.size())

# print(iv3_model)
# for params in iv3_model.parameters():
#     for param in params:
#         print(param.shape)

# print(iv3_model.parameters()[0][0])
# iv3_model = c.torch_to_pytorch(legacymodel=iv3_model)
model.eval()
fake = model(z)
voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))

np_fake = fake.detach().numpy()


def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')


def center_of_mass(voxels, threshold=0.1):
    """ Calculate the center of mass for the current object.
    Voxels with occupancy less than threshold are ignored
    """
    assert voxels.ndim == 3
    center = [0] * 3
    voxels_filtered = np.copy(voxels)
    voxels_filtered[voxels_filtered < threshold] = 0

    total = voxels_filtered.sum()
    if total == 0:
        print
        'threshold too high for current object.'
        return [length / 2 for length in voxels.shape]

        # calculate center of mass
    center[0] = np.multiply(voxels_filtered.sum(1).sum(1), np.arange(voxels.shape[0])).sum() / total
    center[1] = np.multiply(voxels_filtered.sum(0).sum(1), np.arange(voxels.shape[1])).sum() / total
    center[2] = np.multiply(voxels_filtered.sum(0).sum(0), np.arange(voxels.shape[2])).sum() / total

    return center


# visualization(voxels, 0.4, title=None, uniform_size=1, use_colormap=False, angle=0)

threshold = 0.2
small_vals = voxels < threshold
voxels[small_vals] = 0
voxels[~small_vals] = 1
center = center_of_mass(voxels, threshold=threshold)
print(center)

for i in range(4):
    front = voxels[:, :, i * 16:(i + 1) * 16].sum(axis=2) / 16
    small_vals = front < 0.01
    front[small_vals] = 0
    front[~small_vals] = 1
    print(front.shape)

    # front_filter = front > 255.0
    # front[front_filter] = 255
    # front /= max_val
    # show_image(front)
    plt.imshow(front, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

for i in range(4):
    front = voxels[:, i * 16:(i + 1) * 16, :].sum(axis=1) / 16
    small_vals = front < 0.01
    front[small_vals] = 0
    front[~small_vals] = 1
    print(front.shape)

    # front_filter = front > 255.0
    # front[front_filter] = 255
    # front /= max_val
    # show_image(front)
    plt.imshow(front.T, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

for i in range(4):
    front = voxels[i * 16:(i + 1) * 16, :, :].sum(axis=0) / 16
    small_vals = front < 0.01
    front[small_vals] = 0
    front[~small_vals] = 1
    print(front.shape)

    # front_filter = front > 255.0
    # front[front_filter] = 255
    # front /= max_val
    # show_image(front)
    plt.imshow(front.T, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

front = cv2.cvtColor(front, cv2.COLOR_GRAY2BGR)
cv2.imwrite("front.jpg", front)
