import cv2
import numpy as np
import os
import torch
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import load_model
from test_funcs.util_vtk import visualization
from test_funcs.util import downsample

# load data
sketch_path = r"data/sketches2"
model_path = r"data/models/mappingNN2"

# load InceptionV3 iv3_model
iv3_input = (139, 139, 3)
print("Loading Inception V3 Model")
iv3_model = load_model('inceptionv3.h5')
# load mapping model
print("Loading Mapping Model")
mapping_model = load_model(model_path)

# load GAN model
basepath = r"models_cpu/"
filepath = "chair_G_cpu"
gan_model = torch.load(basepath + filepath)

# get sketch
n_sketches = len(os.listdir(sketch_path))
sketch_num = np.random.randint(0, n_sketches - 1)

image_path = sketch_path + "/sketch" + str(sketch_num) + ".png"
image = cv2.imread(image_path)
cv2.imshow("Sketch", image)
cv2.waitKey(0)
cv2.destroyWindow('image')

# get sketch bottleneck features
iv3_input = (139, 139, 3)
images = np.empty((1, iv3_input[0], iv3_input[1], iv3_input[2]))
resized = cv2.resize(image.copy(), (iv3_input[0], iv3_input[1]))
resized = preprocess_input(resized)
images[0] = resized.copy()

# extract bottleneck features
features = iv3_model.predict(images, batch_size=1, verbose=1)

# predict latent vector
latent_vec = mapping_model.predict(features, batch_size=1)[0]
latent_vec = torch.Tensor(latent_vec)
latent_vec = latent_vec.view(1, -1, 1, 1, 1)

# generate shape
gan_model.eval()
fake = gan_model(latent_vec)
np_fake = fake.detach().numpy()
voxels = np.reshape(fake.detach().numpy(), (64, 64, 64))

voxels = downsample(voxels, 4, method='max')
visualization(voxels, 0.5, title=None, uniform_size=1, use_colormap=False, angle=0.3)
