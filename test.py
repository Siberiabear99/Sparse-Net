import torch
import torch.nn as nn
import numpy as np
#import tifffile
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import skimage
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from datetime import datetime

class DoubleConv(nn.Module):
    """
    Convolution 3 x 3
    => 
    [BN] 
    => 
    Leaky ReLU
    Convolution 3 x 3 
    => 
    [BN] 
    => 
    Leaky ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),#AdaptiveAvgPool2d
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        return self.conv(x)


class AUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(AUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)             
        self.down4 = Down(512, 1024 // 2)

        self.up1 = Up(1024, 512 // 2, bilinear)
        self.up2 = Up(512, 256 // 2, bilinear)
        self.up3 = Up(256, 128 // 2, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
    
# 规范化函数
def normalize_(input_img):
    img = np.array(input_img).astype('float32')
    img = img - np.min(img)
    img = img / np.max(img)
    return img

# 预测函数
def predict_(model, img_path, save_path, fname,ifGPU=True):
    device = torch.device("cuda" if torch.cuda.is_available() and ifGPU else "cpu")

    # 读取图像数据
    image_data = imread((img_path)+'/' + fname)
    try:
        t, x, y = image_data.shape  # t为帧数，x和y为图像的高和宽
        image_data = image_data.astype('float32')

        # 创建一个四维的zeros数组，后两个维度为一帧图像的xy，前两个维度变为1, t
        test_pred_np = np.zeros((t, x, y), dtype=np.float32)
    except ValueError:
        t = 1
        x, y = image_data.shape
        image_data = image_data.astype('float32')
        image_data = np.expand_dims(image_data, axis=0)  # 添加帧维度
        test_pred_np = np.zeros((t, x, y), dtype=np.float32)
    # 处理每一帧图像
    for frame in range(t):
        # 取出单帧图像
        single_frame = image_data[frame, :, :]
        # 规范化
        single_frame = normalize_(single_frame)
        single_frame = single_frame.astype('float32')
        single_frame = single_frame.reshape(1,x,y)
        # 增加维度并转换为张量
        #datatensor = torch.from_numpy(single_frame).unsqueeze(0).unsqueeze(0).to(device)
        imsize = (1,1,x,y)
        imagA = np.zeros(imsize)
        imagA[0,:,:,:] = single_frame
        imagA = torch.from_numpy(imagA)
        imagA = imagA.to(device, dtype = torch.float32)
        with torch.no_grad():
            # 模型预测
            test_pred = model(imagA)
            # 移除批次和通道维度，并将结果转换回numpy
            #test_pred = test_pred.squeeze().cpu().numpy()
        test_pred = test_pred.to(torch.device("cpu"))
        test_pred = test_pred.numpy()
        test_pred = normalize_(test_pred)
        #plt.imshow(test_pred[0,0,:,:])
        #plt.show()
        # 保存输出结果到数组中
        test_pred_np[frame, :, :] = test_pred
        if frame % 100 == 0:
            print('predict %d images'% frame)

    # 保存整个数组为一个TIFF文件
    os.makedirs(save_path, exist_ok=True)
    imsave(os.path.join(save_path, '50_'+"%s"%(fname) ), test_pred_np)

def main():

    torch.cuda.empty_cache()

    img_path = 'D:/SparseNet-main/rawdata'  # 图像路径
    save_path = 'D:/SparseNet-main/predictions'  # 保存路径
    model_path = 'D:/SparseNet-main/models/model_4_17_full.pth'  # 模型路径
    fname = 'Substack (1-300).tif'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = AUnet(n_channels = 1, n_classes = 1, bilinear=True).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4, betas = (0.5, 0.999))
    #checkpoint = torch.load(model_path,map_location=device,weights_only=True)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(torch.load(model_path,map_location = device,weights_only = True))
    #model = torch.load(model_path,map_location=device)
    model.eval()
    #print(model.state_dict())
    # 预测并保存结果
    start_time = datetime.now()
    predict_(model, img_path, save_path, fname,ifGPU=True)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"程序运行耗时：{elapsed_time}")
    print('OK')
    

if __name__ == '__main__':
    main()