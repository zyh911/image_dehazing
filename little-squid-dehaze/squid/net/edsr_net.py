#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Implementation of AOD-Net in ICCV2017
# Created by Zhao Yuhang in Meitu.

import os 
import torch
import torch.nn as nn
from torch.autograd import Variable

class EDSR_Net(nn.Module):
	def __init__(self):
		super(EDSR_Net, self).__init__()
		self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
		self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu3 = nn.ReLU(inplace=True)
		self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv4_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu4 = nn.ReLU(inplace=True)
		self.conv4_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv5_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu5 = nn.ReLU(inplace=True)
		self.conv5_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv6_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu6 = nn.ReLU(inplace=True)
		self.conv6_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv7_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu7 = nn.ReLU(inplace=True)
		self.conv7_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv8_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.relu8 = nn.ReLU(inplace=True)
		self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

		self.conv9 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

		self.clip = nn.ReLU(inplace=True)

		self.weights_init()

	def forward(self, x, target=None):
		x0_0 = x
		x0 = self.conv0(x0_0)
		x1_0 = x0
		x1 = self.conv1_1(x1_0)
		x1 = self.relu1(x1)
		x1 = self.conv1_2(x1)
		x2_0 = x1 + x1_0
		x2 = self.conv2_1(x2_0)
		x2 = self.relu2(x2)
		x2 = self.conv2_2(x2)
		x3_0 = x2 + x2_0
		x3 = self.conv3_1(x3_0)
		x3 = self.relu3(x3)
		x3 = self.conv3_2(x3)
		x4_0 = x3 + x3_0
		x4 = self.conv4_1(x4_0)
		x4 = self.relu4(x4)
		x4 = self.conv4_2(x4)
		x5_0 = x4 + x4_0
		x5 = self.conv5_1(x5_0)
		x5 = self.relu5(x5)
		x5 = self.conv5_2(x5)
		x6_0 = x5 + x5_0
		x6 = self.conv6_1(x6_0)
		x6 = self.relu6(x6)
		x6 = self.conv6_2(x6)
		x7_0 = x6 + x6_0
		x7 = self.conv7_1(x7_0)
		x7 = self.relu7(x7)
		x7 = self.conv7_2(x7)
		x8_0 = x7 + x7_0
		x8 = self.conv8_1(x8_0)
		x8 = self.relu8(x8)
		x8 = self.conv8_2(x8)
		
		x9_0 = x8
		x9 = self.conv9(x9_0)
		
		out = x9
		
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

if __name__ == '__main__':
	model = EDSR_Net()
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

