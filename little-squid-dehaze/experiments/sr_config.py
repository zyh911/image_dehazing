#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
sys.path.insert(0, '../../')


# =================== config for train flow ============================================================================
DESC = "L1 "
experiment_name = os.path.splitext(__file__.split('/')[-1])[0]

# BASE_ROOT = '/3T/sr_group'   # 1080a
BASE_ROOT = '/root/group-super-resolution'  # server

TRAIN_ROOT      = '/3T/train_tasks/'+experiment_name

MODEL_FOLDER        = os.path.join(TRAIN_ROOT, 'models')
TRAIN_OUT_FOLDER    = os.path.join(TRAIN_ROOT, 'train_out')
PEEK_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'peek_out')
TEST_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'test_out')

DATASET_DIR = '/3T/images'
DATASET_ID = 'dataset20171030_gauss_noise03_random_ds_random_kernel_jpeg'
DATASET_TXT_DIR = './imgs/'

IMAGE_SITE_URL      = 'http://172.16.3.247:8000/image-site/dataset/{dataset_name}?page=1&size=50'
IMAGE_SITE_DATA_DIR = '/3T/images'

peek_images = ['./imgs/44.png',  './imgs/36.png']
test_input_dir = "/3T/images/ftt-png/"


GPU_ID = 0
epochs = 4
batch_size = 16
start_epoch = 1 
save_snapshot_interval_epoch = 2
peek_interval_epoch = 1
save_train_hr_interval_epoch = 1
loss_average_win_size = 2 
validate_interval_epoch = 1 
plot_loss_start_epoch = 1 
only_validate = False  #

from visdom import Visdom
vis = Visdom(server='http://172.18.11.16', port=8097)


# =================== config for model and dataset =====================================================================
from squid.data import RandomCropPhoto2PhotoData
from squid.model import SuperviseModel
import torch
import torch.nn as nn
from squid.loss import VGGLoss
from squid.net import SrResnet

hr_size = 128

target_net = SrResnet()

model = SuperviseModel({
    'net': target_net, 
    'optimizer': torch.optim.Adam([{'name':'net_params', 'params':target_net.parameters(), 'base_lr':1e-3}], betas=(0.9, 0.999), weight_decay=0.0005),
    'lr_step_ratio': 0.5,
    'lr_step_size': 500,

    'supervise':{
        'out':  {'L1_loss': {'obj': nn.L1Loss(size_average=True),  'factor':0.1, 'weight': 1.0}}, 
    },
    'metrics': {}
     
})

train_dataset = RandomCropPhoto2PhotoData({
            'crop_size': hr_size,
            'crop_stride': 2,
            'data_root': DATASET_DIR,
            'desc_file_path': os.path.join(DATASET_TXT_DIR, DATASET_ID, 'train.txt'),
})

valid_dataset = RandomCropPhoto2PhotoData({
            'crop_size': hr_size,
            'crop_stride': 2,
            'data_root': DATASET_DIR,
            'desc_file_path': os.path.join(DATASET_TXT_DIR, DATASET_ID, 'val.txt'),
})


