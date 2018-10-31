from pyDOE import lhs
import torch
import numpy as np
from utils.util_vtk import *
from utils.util import *
from utils.sketchify import sketchify

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # we're dealing with cpu model

# Load Model
model_basepath = r"models_cpu/"
model_filepath = "chair_G_cpu"
model = torch.load(model_basepath + model_filepath)

# Data Folder
data_folder = r"data/"
sketch_folder = r"sketches/"
sketch_filename = "sketch"
img_ext = ".png"
samples_filename = "samples.npy"

sketch_fp = data_folder + sketch_folder + sketch_filename
samples_fp = data_folder + samples_filename

# Parameters
nsamples = 10
ndim = 200
downsampling = 4
image_size = 128

# Latin Hypercube Sampling
samples = lhs(ndim, samples=nsamples) * 0.5 - 0.5
np.save(samples_fp, samples)

sample_num = 0
for latent_vec in samples:
    z = torch.Tensor(latent_vec)
    z = z.view(1, -1, 1, 1, 1)
    fake = model(z)
    np_fake = fake.detach().numpy()
    voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))
    voxels = downsample(voxels, 4, method='max')
    path = sketch_fp + str(sample_num) + img_ext
    visualization(voxels, 0.5, title=None, uniform_size=1, use_colormap=False, angle=0.3, filename=path)
    sketchify(path, path.split(".")[0] + "bis." + path.split(".")[1])
    sample_num += 1

# Write Samples Data
# f = h5py.File(samples_fp, "w")
# f.create_dataset('samples', data=samples)
# f.close()
