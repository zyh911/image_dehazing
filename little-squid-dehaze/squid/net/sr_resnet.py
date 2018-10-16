#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Implementation of https://arxiv.org/pdf/1512.03385.pdf.
# See section 4.2 for model architecture on CIFAR-10.
# Some part of the code was referenced below.
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import os
import torch 
import torch.nn as nn
from torch.autograd import Variable


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, num_features=64, is_biased=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=is_biased)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=is_biased)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += x 
        return out


class SrResnet(nn.Module):
    def __init__(self, in_channel=3, num_blocks=16, num_features=64, is_biased=True, init_state_path=None):
        super(SrResnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, num_features, kernel_size=3, stride=1, padding=1, bias=is_biased)
        self.relu1 = nn.ReLU(True)

        self.blocks = self.make_blocks_(num_blocks)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=is_biased)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(num_features, in_channel, kernel_size=3, stride=1, padding=1, bias=is_biased)
        self.weights_init()
        if init_state_path:
            checkpoint = torch.load(init_state_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint)


    def make_blocks_(self, num):
        layers = []
        for i in range(num):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        # stage 1
        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)

        # stage 2
        blocks_out = self.blocks(relu1)
        conv2 = self.conv2(blocks_out)
        relu2 = self.relu2(conv2)

        out2 = relu2 + relu1
        out = self.conv3(out2)

        if target is  not None:
            pairs = {'out': (out, target)} 
            return pairs, self.exports(x, out, target)
        else:
            return self.exports(x, out, target)

    def exports(self, x, output, target):
        result = {'input': x, 'output': output}
        if target is not None:
            result['target'] = target
        return result


    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
