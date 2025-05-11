import numpy as np
#import tifffile
import os
import skimage
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

class data_generator():
    def __init__(self, img_path, save_path,img_res = (128, 128),signal_intensity = 0.13):
        self.img_path = img_path
        self.save_path = save_path
        self.img_res = img_res
        self.signal_intensity = signal_intensity

    def data_G(self):
        image_data = imread((self.img_path)+'/*.tif' )
        i = 0
        os.makedirs(self.save_path, exist_ok=True)
        try:
            t, x, y = image_data.shape  
            image_data = image_data.astype('float32')
            num = int(x/self.img_res[0])
        except ValueError:
            t = 1
            x, y = image_data.shape
            image_data = image_data.astype('float32')
            image_data = np.expand_dims(image_data, axis=0)  # 添加帧维度
            num = int(x/self.img_res[0])
        for frame in range(t):
            single_frame = image_data[frame, :, :]
            sparse_part = np.zeros_like(single_frame)
            zero_tensor = np.array(0.0,  dtype=single_frame.dtype)
            sparse_part[:,:] = np.where(single_frame[:,:] < self.signal_intensity, zero_tensor, single_frame[:,:])
            for j in range(num):
                for k in range(num):
                    data = np.zeros((self.img_res[0],2*self.img_res[0]))
                    data[:,0:self.img_res[0]] = single_frame[j*self.img_res[0]:(j+1)*self.img_res[0],k*self.img_res[0]:(k+1)*self.img_res[0]]
                    data[:,self.img_res[0]:2*self.img_res[0]] = sparse_part[j*self.img_res[0]:(j+1)*self.img_res[0],k*self.img_res[0]:(k+1)*self.img_res[0]]
                    data = data.astype('float32')
                    imsave(os.path.join(self.save_path, f'{i}.tif' ), data)
                    i += 1
            if frame % 10 == 0:
                print('create' , f'{frame*(num)**2}' , ' pic')
                



            




