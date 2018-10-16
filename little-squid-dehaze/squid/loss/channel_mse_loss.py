#!/usr/bin/env python
# -*- coding:utf-8 -*-
#---------------------------------------------
# Channel weighted MSE loss for dehazing task
# Created by Jiangfan Deng <djf@meitu.com>
#---------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np


class ChannelMSELoss(nn.Module):
    """Channel MSE Loss"""
    def __init__(self, size_average=True, reduce=True, channel_weights=None):
        super(ChannelMSELoss, self).__init__()
        self.reduce = reduce
        self.size_average = size_average
        self.channel_weights = channel_weights
        if self.channel_weights is not None:
            assert len(self.channel_weights) == 3,\
                'error: channel weights must be 3'

    def forward(self, input, target):
        # _assert_no_grad(target)
        if self.channel_weights is None:
            return F.mse_loss(input, target, size_average=self.size_average, reduce=self.reduce)
        else:
            loss_1 = F.mse_loss(input[:,0,:,:], target[:,0,:,:], size_average=self.size_average, reduce=self.reduce)
            loss_2 = F.mse_loss(input[:,1,:,:], target[:,1,:,:], size_average=self.size_average, reduce=self.reduce)
            loss_3 = F.mse_loss(input[:,2,:,:], target[:,2,:,:], size_average=self.size_average, reduce=self.reduce)
            loss = self.channel_weights[0] * loss_1 + self.channel_weights[1] * loss_2 + self.channel_weights[2] * loss_3
        return loss
