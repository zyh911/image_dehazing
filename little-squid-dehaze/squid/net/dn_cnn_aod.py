#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Created by Zhao Yuhang in Meitu.

import os 
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class DnCnn_AOD(nn.Module):
    """Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"""
    def __init__(self, image_c=3, feature_num=64, layer_num=10):
        super(DnCnn_AOD, self).__init__()
        self.feature_num = feature_num
        self.layer_num = layer_num
        self.conv1 =  nn.Conv2d(image_c, self.feature_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.feature_num)
        self.relu1 = nn.ReLU()
        self.intermediate_layers = self.make_layer()
        self.reconstruct = nn.Conv2d(self.feature_num, image_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.clip = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.weight.data.normal_(0, 0.0001)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def make_layer(self):
        layers = []
        for i in range(self.layer_num):
            layers.append(nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(self.feature_num))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.intermediate_layers(out)
        out = self.reconstruct(out)
        out = out + x
        out = out * x - out + 1
        out = self.clip(out)
        
        if target is not None:
            pairs = {'out': (out, target)}
            return pairs, self.exports(x, out, target)
        else:
            return self.exports(x, out, target)

    def exports(self, x, output, target):
        result = {'input': x, 'output': output}
        if target is not None:
            result['target'] = target
        return result


