import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
import torch
from utils.Sparsetrain_V1_1 import Sparse
import datetime

torch.cuda.empty_cache()
now = datetime.datetime.now()
formatted_now = now.strftime("%Y%m%d_%H%M%S")

dataset_name = '80'
epochs = 150
train_batch_size = 64
test_batch_size = 1
lam = 0.0884

prefix = 'Neural_125_'
tests_name = (prefix + ('_%s_EPOCH%d_BS%d'%(dataset_name, epochs, train_batch_size))+f'_{formatted_now}_')


os.makedirs((('images/%s'%(tests_name))), exist_ok=True)
os.makedirs((('images/%s/weights'%(tests_name))), exist_ok=True)
os.makedirs((('images/%s/images'%(tests_name))), exist_ok=True)

sparsenet = Sparse(dataset_name = dataset_name, tests_name = tests_name, epochs = epochs, train_batch_size = train_batch_size, 
                     test_batch_size = test_batch_size,lam = lam,ifdetach = False)
sparsenet.train()
