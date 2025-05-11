import numpy as np
import os
from skimage.io import imread, imsave

def read_square(position, length, image):
    # 提取指定位置和大小的区域
    extracted_region = image[position[0]:position[0] + length, position[1]:position[1] + length]
    return extracted_region

def gb(low_signal_intensity, high_signal_intensity, img_path, raw_path, save_path, fname, fname2, length):
    # 读取图像数据
    image_data = imread(os.path.join(img_path, fname))
    image_data_raw = imread(os.path.join(raw_path, fname2))
    try:
        t, x, y = image_data.shape
        image_data = image_data.astype('float32')
        test_pred_np = np.zeros((t, x, y), dtype=np.float32)
    except ValueError:
        t = 1
        x, y = image_data.shape
        image_data = image_data.astype('float32')
        image_data = np.expand_dims(image_data, axis=0)
        test_pred_np = np.zeros((t, x, y), dtype=np.float32)

    # 处理每一帧图像
    for frame in range(t):
        single_frame = image_data[frame, :, :]
        single_frame_raw = image_data_raw[frame, :, :]
        sparse_part = np.zeros((x, y), dtype=single_frame.dtype)

        num = int(x / length)
        for i in range(num):
            for j in range(num):
                position = [i * length, j * length]
                square = read_square(position, length, single_frame)
                square_raw = read_square(position, length, single_frame_raw)

                flattened_array = square.flatten()
                sorted_array = np.sort(flattened_array)
                count = np.sum(sorted_array > high_signal_intensity)
                rou = count / (length ** 2)
                if rou <0.001:
                    threshold_index = int(len(sorted_array) * (0.98))
                else:
                    threshold_index = int(len(sorted_array) * (1 - 1 * rou / 1))

                # 确保 threshold_index 不会超出范围
                if threshold_index >= len(sorted_array):
                    threshold_index = len(sorted_array) - 1

                threshold_value = sorted_array[threshold_index]

                zero_square = np.zeros((length, length))
                zero_square[:, :] = np.where(np.logical_or(square[:, :] < threshold_value, square[:, :] < low_signal_intensity), 0.0, square_raw[:, :])

                sparse_part[position[0]:position[0] + length, position[1]:position[1] + length] = zero_square[:, :]

        test_pred_np[frame, :, :] = sparse_part

        if frame % 100 == 0:
            print('generate %d images' % frame)

    os.makedirs(save_path, exist_ok=True)
    imsave(os.path.join(save_path, 'C_' + fname), test_pred_np)

def main():
    raw_path = 'D:/SparseNet-main/rawdata'
    img_path = 'D:/SparseNet-main/predictions'
    save_path = 'D:/SparseNet-main/correction'
    fname = 'lam0_8_80_Substack (1-300).tif'
    fname2 = 'Substack (1-300).tif'
    low_signal_intensity = 0.03
    high_signal_intensity = 0.1
    length = 64
    gb(low_signal_intensity=low_signal_intensity, high_signal_intensity=high_signal_intensity, img_path=img_path, raw_path=raw_path, save_path=save_path, fname=fname, fname2=fname2, length=length)
    print('OK')

if __name__ == '__main__':
    main()