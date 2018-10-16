#!/usr/bin/env python
# -*- coding:utf-8 -*-
# changed gpus
import os
import sys
sys.path.insert(0, '../../') # to be changed


# =================== config for train flow ============================================================================
DESC = 'MSE' # to be changed
experiment_name = os.path.splitext(__file__.split('/')[-1])[0]

BASE_ROOT = '/root/group-competition/xiamen'  # server

TRAIN_ROOT = BASE_ROOT + '/train_tasks/' + experiment_name

MODEL_FOLDER        = os.path.join(TRAIN_ROOT, 'models')
TRAIN_OUT_FOLDER    = os.path.join(TRAIN_ROOT, 'train_out')
PEEK_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'peek_out')
TEST_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'test_out')

DATASET_DIR = BASE_ROOT + '/ITS/ITS'
DATASET_ID = 'aod_txt'
DATASET_TXT_DIR = BASE_ROOT + '/ITS/ITS'

# IMAGE_SITE_URL      = 'http://172.16.3.247:8000/image-site/dataset/{dataset_name}?page=1&size=50'
# IMAGE_SITE_DATA_DIR = '/root/zyh3/ITS/ITS/train'

peek_images = [BASE_ROOT + '/ITS/ITS/train/ITS_haze/0499_05_0.8353.png',  BASE_ROOT + '/ITS/ITS/train/ITS_haze/0488_05_0.9738.png']
test_input_dir = BASE_ROOT + '/SOTS/SOTS/indoor/nyuhaze500/hazy'

GPU_ID = None
epochs = 20
batch_size = 16
start_epoch = 1 
save_snapshot_interval_epoch = 2
peek_interval_epoch = 1
save_train_hr_interval_epoch = 1
loss_average_win_size = 2 
validate_interval_epoch = 1 
validate_batch_size = 1 
plot_loss_start_epoch = 1 
only_validate = False  #

from visdom import Visdom
vis = Visdom(server='http://127.0.0.1', port=8097)


# =================== config for model and dataset =====================================================================
from squid.data import Photo2PhotoData
from squid.data import RandomCropPhoto2PhotoData
from squid.model import SuperviseModel
import torch
import torch.nn as nn
from squid.loss import VGGLoss
from squid.net import AODNet

hr_size = (407, 541)
target_net = AODNet()
target_net = nn.DataParallel(target_net).cuda()

model = SuperviseModel({
    'net': target_net, 
    'optimizer': torch.optim.Adam([{'name':'net_params', 'params':target_net.parameters(), 'base_lr':1e-4}], betas=(0.9, 0.999), weight_decay=0.0005),
    'lr_step_ratio': 0.5,
    'lr_step_size': 5,

    'supervise':{
        'out':  {'MSE_loss': {'obj': nn.MSELoss(size_average=True),  'factor':0.1, 'weight': 1.0}}, 
    },
    'metrics': {}
     
})

train_dataset = RandomCropPhoto2PhotoData({
            'crop_size': hr_size,
            'crop_stride': 2,
            'data_root': DATASET_DIR,
            'desc_file_path': os.path.join(DATASET_TXT_DIR, DATASET_ID, 'train.txt'),
})

valid_dataset = Photo2PhotoData({
            'data_root': DATASET_DIR,
            'desc_file_path': os.path.join(DATASET_TXT_DIR, DATASET_ID, 'val.txt'),
})


