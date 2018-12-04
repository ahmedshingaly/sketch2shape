import torch as torch
import sys
sys.path.append(r"C:\Users\renau\Dropbox (MIT)\1_PhD\Code\Machine Learning\6s198_project\sketch2shape")
from torch.legacy import nn as nn
import test_funcs.convertLegacy as c
import numpy as np
from test_funcs.util_vtk import visualization
from test_funcs.util import downsample
import vtk

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # we're dealing with cpu iv3_model

# load model
basepath = r"../models_cpu/"
filepath = "chair_G_cpu"
model = torch.load(basepath + filepath)

# generate random latent vectors
batch_size = 1
latent_size = 200
n_dim = 100  # number of dimensions not kept fixed
random_subvec = np.random.random(n_dim)
latent_vec = np.zeros(latent_size)
latent_vec[n_dim:2*n_dim] = random_subvec

z = torch.randn(batch_size, latent_size, device=device) * 1
# z = torch.Tensor(latent_vec)
print(z)
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
np_fake = fake.detach().numpy()
voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))

voxels = downsample(voxels, 2, method='max')
visualization(voxels, 0.5, title=None, uniform_size=1, use_colormap=False, angle=0.3)
