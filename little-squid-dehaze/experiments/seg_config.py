#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys

DESC = "L1 "
experiment_name = os.path.splitext(__file__.split('/')[-1])[0]


# =================== dirs =====================================================================
TRAIN_ROOT      = '/3T/train_tasks/'+experiment_name
MODEL_FOLDER        = os.path.join(TRAIN_ROOT, 'models')
TRAIN_OUT_FOLDER    = os.path.join(TRAIN_ROOT, 'train_out')
PEEK_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'peek_out')
TEST_OUT_FOLDER     = os.path.join(TRAIN_ROOT, 'test_out')

IMAGE_SITE_URL      = 'http://172.16.3.247:8000/image-site/dataset/{dataset_name}?page=1&size=50'
peek_images = ['/3T/images/Asian_FacialSegementation_3000_640x480/2158_IMG_3153_face.jpg',
               '/3T/images/Asian_FacialSegementation_3000_640x480/1024_IMG_1068_face.jpg',]
test_input_dir = "/3T/xcb/pytorch-srgan/examples/super_resolution/seg_test" 

IMAGE_SITE_URL      = 'http://172.16.3.247:8000/image-site/dataset/{dataset_name}?page=1&size=50'
IMAGE_SITE_DATA_DIR = '/3T/images'
# =================== params =====================================================================

GPU_ID = 0
epochs = 4
batch_size = 20
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

# =================== net and model =====================================================================
import torch
import torch.nn as nn
from squid.net import ICNet
from squid.model  import SuperviseModel
from squid.metric import IouScore
from squid.metric import AccScore
from squid.data import Photo2MaskData

target_net = ICNet(nclass=15)

model = SuperviseModel({
    'net': target_net, 
    'optimizer': torch.optim.Adam([{'name':'net_params', 'params':target_net.parameters(), 'base_lr':1e-3}], betas=(0.9, 0.999), weight_decay=0.0005),
    'lr_step_ratio': 0.5,
    'lr_step_size': 500,
    'supervise':{
        'out1': {'cross_entrypy': {'obj': nn.CrossEntropyLoss(size_average=True),  'factor':1.0, 'weight': 1.0}}, 
        'out2': {'cross_entrypy': {'obj': nn.CrossEntropyLoss(size_average=True),  'factor':1.0, 'weight': 1.0}}, 
        'out3': {'cross_entrypy': {'obj': nn.CrossEntropyLoss(size_average=True),  'factor':1.0, 'weight': 1.0}}, 
        'out4': {'cross_entrypy': {'obj': nn.CrossEntropyLoss(size_average=True),  'factor':1.0, 'weight': 1.0}}, 
    },
    'metrics':{
        'mask_out': {'iou': {'obj': IouScore(nclass=15)}, 
                     'acc': {'obj': AccScore()}
                    }, 
    },
})

# =================== dataset =====================================================================
train_dataset = Photo2MaskData({
            'desc_file_path':'/3T/xcb/pytorch-srgan/examples/face_parse/txt/train_for_dev.txt',
})

valid_dataset = Photo2MaskData({
            'desc_file_path':'/3T/xcb/pytorch-srgan/examples/face_parse/txt/valid_for_dev.txt',
})
