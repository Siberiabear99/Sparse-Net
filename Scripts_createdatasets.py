import os
import numpy as np
from glob import glob
from utils.createdatasets import data_generator





img_path = ('D:/SparseNet-main/rawdata')
save_path = ('D:/SparseNet-main/datasets')
os.makedirs(save_path, exist_ok=True)

signal_intensity = 0.21

d = data_generator(img_path = img_path, save_path = save_path,img_res = (128, 128),signal_intensity = signal_intensity)
d.data_G()
print('OK')