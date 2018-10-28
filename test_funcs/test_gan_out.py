import torch as torch
from torch.legacy import nn as nn
import test_funcs.convertLegacy as c
import numpy as np
from test_funcs.util_vtk import visualization
from test_funcs.util import downsample
import vtk

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'  # we're dealing with cpu model

# load model
basepath = r"../models_cpu/"
filepath = "chair_G_cpu"

model = torch.load(basepath + filepath)

# generate random latent vectors
batch_size = 1
latent_size = 200

z = torch.randn(batch_size, latent_size, device=device)
z = z.view(1, -1, 1, 1, 1)
print(z.size())

# print(model)
# for params in model.parameters():
#     for param in params:
#         print(param.shape)

# print(model.parameters()[0][0])
model = c.torch_to_pytorch(legacymodel=model)
model.eval()
fake = model(z)
np_fake = fake.detach().numpy()
voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))

voxels = downsample(voxels, 2, method='max')
visualization(voxels, 0.5, title=None, uniform_size=1, use_colormap=False, angle=0.3)



