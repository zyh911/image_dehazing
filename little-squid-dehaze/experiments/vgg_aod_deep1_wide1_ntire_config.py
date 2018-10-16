#!/usr/bin/env python
# -*- coding:utf-8 -*-
# changed gpus
import os
import sys
sys.path.insert(0, '../../') # to be changed


# =================== config for train flow ============================================================================
DESC = 'MSE' # to be changed
experiment_name = os.path.splitext(__file__.split('/')[-1])[0]

INDOOR_OR_OUTDOOR_DIR = '/indoor'
# INDOOR_OR_OUTDOOR_DIR = '/outdoor'

PERSONAL_TASKS = '/zyh3/train_tasks' # personal
BASE_ROOT = '/root/group-competition'  # base

TRAIN_ROOT = BASE_ROOT + PERSONAL_TASKS + '/' + experiment_name

MODEL_FOLDER        = os.path.join(TRAIN_ROOT, 'models')
TRAIN_OUT_FOLDER    = os.path.join(TRAIN_ROOT, 'train_out')
PEEK_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'peek_out')
TEST_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'test_out')

DATASET_DIR = '/root/train_val_last_set'
DATASET_ID = 'aod_txt'
DATASET_TXT_DIR = '/root/train_val_last_set'

# IMAGE_SITE_URL      = 'http://172.16.3.247:8000/image-site/dataset/{dataset_name}?page=1&size=50'
# IMAGE_SITE_DATA_DIR = '/root/zyh3/ITS/ITS/train'

peek_images = ['/root/train_val_last_set/train/haze/0.png',  '/root/train_val_last_set/train/haze/100.png']
test_input_dir = BASE_ROOT + '/data/dehaze/NTIRE' + INDOOR_OR_OUTDOOR_DIR + '/IndoorTestHazy'

GPU_ID = 0
epochs = 10
batch_size = 16
start_epoch = 0
save_snapshot_interval_epoch = 1
peek_interval_epoch = 1
save_train_hr_interval_epoch = 1
loss_average_win_size = 2 
validate_interval_epoch = 1
validate_batch_size = 4
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
from squid.loss import VGGLoss1
from squid.net import AOD_Deep1_Wide1_Net

target_net = AOD_Deep1_Wide1_Net()
# target_net = nn.DataParallel(target_net).cuda()

model = SuperviseModel({
    'net': target_net, 
    'optimizer': torch.optim.Adam([{'name':'net_params', 'params':target_net.parameters(), 'base_lr':1e-4}], betas=(0.9, 0.999), weight_decay=0.0005),
    'lr_step_ratio': 0.5,
    'lr_step_size': 2,

    'supervise':{
        'out': {'MSE_loss': {'obj': nn.MSELoss(size_average=True), 'factor':1.0, 'weight': 1.0}, 
                'VGG_loss': {'obj': VGGLoss1(vgg19_feature_model_path='/root/group-competition/pretrain_model/vgg_tf.pth', out_feature_level=18), 'factor':1e-7, 'weight': 1.0}}, 
    },
    'metrics': {}
     
})

train_dataset = Photo2PhotoData({
            'data_root': DATASET_DIR,
            'desc_file_path': os.path.join(DATASET_TXT_DIR, DATASET_ID, 'train.txt'),
})

valid_dataset = Photo2PhotoData({
            'data_root': DATASET_DIR,
            'desc_file_path': os.path.join(DATASET_TXT_DIR, DATASET_ID, 'val.txt'),
})


