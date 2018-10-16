#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Implementation of AOD-Net in ICCV2017
# Created by Zhao Yuhang in Meitu.

import os 
import torch
import torch.nn as nn
from torch.autograd import Variable

class AOD_Wide1_Net(nn.Module):
	def __init__(self):
		super(AOD_Wide1_Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=1, stride=1)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
		self.relu3 = nn.ReLU(inplace=True)
		self.conv4 = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3)
		self.relu4 = nn.ReLU(inplace=True)
		self.conv5 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
		self.relu5 = nn.ReLU()
		self.clip = nn.ReLU(inplace=True)

	def forward(self, x, target=None):
		x1_0 = x
		x1 = self.conv1(x1_0)
		x1 = self.relu1(x1)
		x2_0 = x1
		x2 = self.conv2(x2_0)
		x2 = self.relu2(x2)
		x3_0 = torch.cat((x1, x2), dim=1)
		x3 = self.conv3(x3_0)
		x3 = self.relu3(x3)
		x4_0 = torch.cat((x2, x3), dim=1)
		x4 = self.conv4(x4_0)
		x4 = self.relu4(x4)
		x5_0 = torch.cat((x1, x2, x3, x4), dim=1)
		x5 = self.conv5(x5_0)
		x5 = self.relu5(x5)
		out = x5 * x - x5 + 1
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


if __name__ == '__main__':
	model = AOD_Wide1_Net()
	x = Variable(torch.ones(4, 3, 100, 100))
	y = Variable(torch.ones(4, 3, 100, 100))
	if torch.cuda.is_available():
		model.cuda()
		x = x.cuda()
		y = y.cuda()
	model.eval()

	out = model(x)
	for key, value in out.items():
		print(key + ':', value.size())

