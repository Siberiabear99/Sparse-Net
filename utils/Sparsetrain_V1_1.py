import os
import random
import skimage
import torch
import datetime
import numpy as np
from glob import glob
from skimage.io import imread, imsave
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .AUnet import AUnet
import matplotlib.pyplot as plt

class Sparse():
    def __init__(self,dataset_name,tests_name,epochs = 100,train_batch_size = 32,test_batch_size = 1,lr = 2e-4,img_res = (128,128),lam = 0.0884,ifdetach = False):
        self.dataset_name = dataset_name
        self.tests_name = tests_name
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AUnet(n_channels = 1, n_classes = 1, bilinear=True).to(self.device)
        self.model2 = AUnet(n_channels = 1, n_classes = 1, bilinear=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas = (0.5, 0.999))
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr = self.lr, betas = (0.5, 0.999))
        self.criterion = torch.nn.L1Loss(reduction='sum')
        self.img_res = img_res
        self.lam = lam
        self.ifdetach = ifdetach
        
    def train(self):
        start_time = datetime.datetime.now() 
        history = []
        test_metrics = []
        lr_ms =[]
        traindata_path = ('D:/SparseNet-main/datasets' )
        path = glob(traindata_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.train_batch_size))
        save_path = (( 'images/%s/images' %(self.tests_name)))

        print(self.device)

        for epoch in range(self.epochs):
            i = 0
            for inputs, labels in self.load_batch(traindata_path): # dataloader 是先shuffle后mini_batch  
                inputs = torch.from_numpy(inputs)
                labels = torch.from_numpy(labels)
                inputs, labels = inputs.to(self.device, dtype = torch.float32), labels.to(self.device, dtype = torch.float32)
                batch_size, channels, height, width = inputs.shape
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()                                
                inputs_pred1 = self.model(inputs)
                inputs_pred2 = self.model2(inputs)

                height_tensor = torch.tensor(height, dtype=torch.float32)
                #lam = 1/torch.sqrt(height_tensor)
                lam = self.lam
                dense = inputs-inputs_pred1
                try:
                    U, S, V = torch.svd_lowrank(dense, q=min(height, width) // 2)
                except torch._C._LinAlgError:
                    print("SVD decomposition failed with svd_lowrank, using torch.svd instead.")
                    U, S, V = torch.svd(dense)
                try:
                    U, S2, V = torch.svd_lowrank(inputs_pred2, q=min(height, width) // 2)
                except torch._C._LinAlgError:
                    print("SVD decomposition failed with svd_lowrank, using torch.svd instead.")
                    U, S2, V = torch.svd(inputs_pred2)

                S = torch.sign(S)*S
                S2 = torch.sign(S2)*S2
                
                loss1_1 = torch.sum(S) + lam * torch.sum(inputs_pred1)
                loss1_2 = self.criterion(inputs-inputs_pred2,inputs_pred1)
                loss2_1 = torch.sum(S2) + lam * torch.sum(inputs-inputs_pred2)
                loss2_2 = self.criterion(inputs-inputs_pred1,inputs_pred2)
                if self.ifdetach == True:
                    loss = loss1_1/loss1_1.detach() + loss1_2/loss1_2.detach()
                    loss2 = loss2_1/loss2_1.detach() + loss2_2/loss2_2.detach()
                    total_loss = loss1_1+loss1_2
                else:
                    loss = loss1_1 + loss1_2
                    loss2 = loss2_1 + loss2_2
                    total_loss = loss.item()

                #loss = self.criterion(inputs_pred1, labels)
                #plt.imshow(labels.detach().cpu().numpy(),cmap = 'gray')
                #plt.show()
                
                historym = np.array(total_loss)
                history.append(historym) 
                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d/%d] [Batch %d/%d] [loss:%f] time:%s" 
                      %(epoch, self.epochs, i, batch_num, total_loss , elapsed_time))
                loss.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()
                self.optimizer2.step()            
                i = i + 1
                 
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            lr_m = np.array(lr)
            lr_ms.append(lr_m) 


            test_path = ( 'D:/SparseNet-main/test' )

            if self.test_batch_size > 0:
                torch.cuda.empty_cache()
                test_pred = self.test(test_path)  
                test_pred2 = self.test2(test_path)
                test_pred = test_pred.to(torch.device("cpu"))
                test_pred2 = test_pred2.to(torch.device("cpu"))
                test_pred = test_pred.numpy()
                test_pred2 = test_pred2.numpy()
                # test_metric = self.caculate_metric_dynamic(test_pred, img_gt_path)
                # test_m = np.array(test_metric)
                # test_metrics.append(test_m) 
                self.saveResult(epoch, save_path, test_pred)
                self.saveResult2(epoch, save_path, test_pred2)
                '''
                if epoch == 1:
                    torch.save(self.model.state_dict(),  ('/images/%s/weights/model_%d_%d_%d.pth')
                               %(self.tests_name, datetime.datetime.now().month, datetime.datetime.now().day, epoch))
                    print(self.model.state_dict())
                '''
                if epoch % 10 == 0:
                    # torch.save(self.model.state_dict(), ( '/images/%s/weights/weights_%d_%d_%d.pth')
                    #            %(self.tests_name,datetime.datetime.now().month, datetime.datetime.now().day,epoch))
                    '''
                    torch.save(
                        {'model_state_dict' : self.model.state_dict(),  
                        'optimizer_state_dict' : self.optimizer.state_dict(),},
                        ('/images/%s/weights/model_%d_%d_%d.pth')
                               %(self.tests_name, datetime.datetime.now().month, datetime.datetime.now().day, epoch)) #gaidong self.model.module.state_dict()
                    '''
                    torch.save(self.model.state_dict(), ( 'images/%s/weights/model_%d_%d_%d.pth')
                                %(self.tests_name,datetime.datetime.now().month, datetime.datetime.now().day,epoch))
                    
        # torch.save(self.model.state_dict(),  ('/images/%s/weights/weights_%d_%d_full.pth')
        #                %(self.tests_name, datetime.datetime.now().month, datetime.datetime.now().day))
        torch.save(self.model.state_dict(),  ('images/%s/weights/model_%d_%d_full.pth')
                       %(self.tests_name, datetime.datetime.now().month, datetime.datetime.now().day)) 
        f1 = open(('images/%s/weights/loss.txt') %(self.tests_name), 'w')
        for i in history:
            f1.write(str(i)+'\r\n')
        f1.close()
        
        f2 = open(('images/%s/weights/lr.txt') %(self.tests_name), 'w')
        for i in lr_ms:
            f2.write(str(i)+'\r\n')
        f2.close()
        
    def test(self, test_path):        
        with torch.no_grad():
            for data in self.load_test_batch(test_path):
                data = torch.from_numpy(data)
                data= data.to(self.device, dtype = torch.float32)
                y_pred = self.model(data)
                return y_pred  
            
    def saveResult(self, epoch, save_path, image_arr):
        for i, item in enumerate(image_arr):
            item = self.normalize(item)
            imsave(os.path.join(save_path, "epoch_%d.tif" %(epoch)), item)

    def saveResult2(self, epoch, save_path, image_arr):
        for i, item in enumerate(image_arr):
            #item = self.normalize(item)
            imsave(os.path.join(save_path, "epoch_%d_b.tif" %(epoch)), item)

    def normalize(self, stack):
        stack = stack.astype('float32')
        stack = stack - np.min(stack)
        stack = stack / np.max(stack)
        return stack  
     
    # def caculate_metric_dynamic(self, img_pred, img_gt_path):    
    #     test_data = SN2NTestset(img_gt_path)
    #     test_loader = DataLoader(test_data, batch_size = self.test_batch_size, shuffle = False)
    #     with torch.no_grad():
    #         for data in self.test_loader:
    #             im_gt = data.numpy()          
    #     im_gt = np.squeeze(self.normalize(im_gt))
    #     img_pred = np.squeeze(self.normalize(img_pred))        
    #     psnr = skimage.metrics.peak_signal_noise_ratio(im_gt, img_pred)
    #     ssim = skimage.metrics.structural_similarity(im_gt, img_pred)
    #     nrmse = skimage.metrics.normalized_root_mse(im_gt, img_pred)
    #     metric = (psnr, ssim, nrmse)
    #     print("=======[PSNR = %f] [SSIM = %f] [NRMSE = %f]=======" %(psnr, ssim, nrmse))
    #     return metric
    
    
    def load_batch(self, traindata_path):  
        path = glob(traindata_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.train_batch_size))
        imsize = (self.train_batch_size, 1, self.img_res[0], self.img_res[1])
        for i in range(batch_num):
            location = np.array(np.random.randint(low = 0, high=len(path), size=(self.train_batch_size, 1), dtype = 'int'))
            for batchsize in range(self.train_batch_size):
            # location = location.tolist()
                batch = []
                batch_tem = path[int(location[batchsize, :])]
                batch.append(batch_tem)
                imgs_As = []
                imgs_Bs = []
                for img in batch:
                    img = imread(img)
                    h, w = img.shape
                    half_w = int(w/2)
                    img_data = img[:, :half_w]
                    img_label = img[:, half_w:]
                    a = np.random.random()
                    b = np.random.random()
                    if a < 0.5:
                        img_data = np.fliplr(img_data)
                        img_label = np.fliplr(img_label)
                    if a > 0.5:
                        img_data = np.flipud(img_data)
                        img_label = np.flipud(img_label)
                    if b < 0.33:
                        img_data = np.rot90(img_data, 1)
                        img_label = np.rot90(img_label, 1)
                    if b > 0.33 and b < 0.66:
                        img_data = np.rot90(img_data, 2)
                        img_label = np.rot90(img_label, 2)
                    if b > 0.66:
                        img_data = np.rot90(img_data, 3)
                        img_label = np.rot90(img_label, 3)
                    if np.max(img_label) != 0:
                        img_label = img_label - np.min(img_label)
                        img_label = img_label / np.max(img_label)
                    else:
                        img_label = img_label
                    img_data = img_data - np.min(img_data)
                    img_data = img_data / np.max(img_data)
                    img_data = img_data.astype('float32')
                    img_label = img_label.astype('float32')
                    img_data = img_data.reshape(1, h, half_w)
                    img_label = img_label.reshape(1, h, half_w)
                    imgs_As.append(img_data)
                    imgs_Bs.append(img_label)
            imgs_As = np.array(imgs_As)
            imgs_Bs = np.array(imgs_Bs)
            yield imgs_As, imgs_Bs
    
    
    def load_test_batch(self, test_path):
        path = glob(test_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.test_batch_size))
        img_tem = imread(path[0])
        h, w = img_tem.shape
        imsize = (self.test_batch_size, 1, h, w)
        imgs_A=np.zeros(imsize)
        for i in range(batch_num):
            batch = path[i*self.test_batch_size:(i+1)*self.test_batch_size]
            for img in batch:
                img = imread(img)
                h, w = img.shape
                img = img - np.min(img)
                img = img / np.max(img)
                img = img.astype('float32')
                img = img.reshape(1, h, w)
               
            imgs_A[i, :, :, :] = img
            yield imgs_A

    def test2(self, test_path):        
        with torch.no_grad():
            for data in self.load_test_batch(test_path):
                data = torch.from_numpy(data)
                data= data.to(self.device, dtype = torch.float32)
                y_pred = self.model2(data)
                return y_pred  
