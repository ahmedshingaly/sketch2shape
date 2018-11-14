import torch
from torch.utils.serialization import load_lua

model_path = r"./models_cpu_non_converted/"
model_file = "car_G_cpu"
full_path = model_path + model_file

model = load_lua(model_file)
