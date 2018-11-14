import numpy as np
import cv2
import os

# Data Folder
sketch_folder = r"data/sketches3/"
sketch_filename = "sketch"
img_ext = ".png"
img_size = (128, 128)
n_rows = 10
n_columns = 20
composite = np.empty((n_rows * img_size[0], n_columns * img_size[1]))
i = 0
for sketch in os.listdir(sketch_folder):
    if i < n_rows * n_columns and sketch.endswith(img_ext):
        path = sketch_folder + sketch
        s = cv2.imread(path)
        s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        row_pos = i // n_columns
        col_pos = i - n_columns * row_pos
        composite[row_pos * img_size[0]:(row_pos + 1) * img_size[0],
        col_pos * img_size[1]:(col_pos + 1) * img_size[1]] = s
        i += 1
cv2.imwrite("composite.png", composite)
