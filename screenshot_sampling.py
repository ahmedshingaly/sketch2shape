from pyDOE import lhs
import torch
import numpy as np
from utils.util_vtk import *
from utils.util import *
from utils.sketchify import sketchify
from time import time
#from sklearn.cluster import DBSCAN


# def filter(voxels, threshold=0.5, distance=10, min_voxels=10):
#     filter_size = voxels > threshold
#     voxels[~filter_size] = 0
#     non_zeros = np.nonzero(voxels)
#     non_zeros = np.vstack(non_zeros).T
#     # print(non_zeros.shape)
#     db = DBSCAN(eps=distance, min_samples=min_voxels)
#     db.fit(non_zeros)
#     (values, counts) = np.unique(db.labels_, return_counts=True)
#     ind = values[np.argmax(counts)]
#     print(ind)
#     # print(ind)
#     # print(np.max(counts))
#     not_retained = non_zeros[db.labels_ != ind]
#     print(not_retained.shape[0])
#     # voxels = np.zeros((64, 64, 64))
#     voxels[not_retained] = 0
#     return voxels


# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # we're dealing with cpu iv3_model

# Load Model
model_basepath = r"models_cpu/"
model_filepath = "chair_G_cpu"
model = torch.load(model_basepath + model_filepath)

# Data Folder
data_folder = r"data/"
sketch_folder = r"screenshotsBT/"
sketch_filename = "screenshot"
img_ext = ".png"
samples_filename = "samples_screenshot_BT.npy"

sketch_fp = data_folder + sketch_folder + sketch_filename
samples_fp = data_folder + samples_filename

# Parameters
nsamples = 10000
ndim = 200
downsampling = 2
image_size = 128
overwrite = True

# Latin Hypercube Sampling
samples = lhs(ndim, samples=nsamples, criterion=None)

np.save(samples_fp, samples)

print(10 * '#' + ' Sampling Started ' + 10 * '#')
tic = time()
sample_num = 0
for latent_vec in samples:
    z = torch.Tensor(latent_vec)
    # z = torch.randn(1, ndim, device=device)*20
    z = z.view(1, -1, 1, 1, 1)
    fake = model(z)
    np_fake = fake.detach().numpy()
    voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))
    if downsampling > 1:
        voxels = downsample(voxels, downsampling, method='mean')
    # voxels = filter(voxels)
    # print(max_connected(voxels, distance=3).shape)
    path = sketch_fp + str(sample_num) + img_ext
    visualization(voxels, 0.3, title=None, uniform_size=1, use_colormap=False, angle=0.3, filename=path)
    # out_path = path
    # if ~overwrite:
    #     out_path = path.split(".")[0] + "bis." + path.split(".")[1]
    # sketchify(path, path, output_dim=(128, 128))
    sample_num += 1
    toc = time()
    print("Sketch " + str(sample_num) + "/" + str(nsamples) + " Sampled | Elapsed Time: " + "{0:.2f}".format(
        toc - tic) + "s | Estimated Remaining Time: " + "{0:.2f}".format(
        (toc - tic) / sample_num * (nsamples - sample_num)) + "s")

    # Write Samples Data
    # f = h5py.File(samples_fp, "w")
    # f.create_dataset('samples', data=samples)
    # f.close()
