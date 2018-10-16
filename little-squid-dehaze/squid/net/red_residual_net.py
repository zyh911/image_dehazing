#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Implementation of AOD-Net in ICCV2017
# Created by Zhao Yuhang in Meitu.

import os 
import torch
import torch.nn as nn
from torch.autograd import Variable

class RED_Residual_Net(nn.Module):
	def __init__(self):
		super(RED_Residual_Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(32)
		self.relu3 = nn.ReLU(inplace=True)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(32)
		self.relu4 = nn.ReLU(inplace=True)
		self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU(inplace=True)

		self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU(inplace=True)
		self.conv7 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU(inplace=True)
		self.conv8 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn8 = nn.BatchNorm2d(32)
		self.relu8 = nn.ReLU(inplace=True)
		self.conv9 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn9 = nn.BatchNorm2d(32)
		self.relu9 = nn.ReLU(inplace=True)
		self.conv10 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
		self.bn10 = nn.BatchNorm2d(3)
		self.relu10 = nn.ReLU(inplace=True)

	def forward(self, x, target=None):
		x1_0 = x
		x1 = self.conv1(x1_0)
		x1 = self.bn1(x1)
		x1 = self.relu1(x1)
		x2_0 = x1
		x2 = self.conv2(x2_0)
		x2 = self.bn2(x2)
		x2 = self.relu2(x2)
		x3_0 = x2
		x3 = self.conv3(x3_0)
		x3 = self.bn3(x3)
		x3 = self.relu3(x3)
		x4_0 = x3
		x4 = self.conv4(x4_0)
		x4 = self.bn4(x4)
		x4 = self.relu4(x4)
		x5_0 = x4
		x5 = self.conv5(x5_0)
		x5 = self.bn5(x5)
		x5 = self.relu5(x5)
		x6_0 = x5
		x6 = self.conv6(x6_0)
		x6 = self.bn6(x6)
		x6 = self.relu6(x6)
		x7_0 = x6 + x4
		x7 = self.conv7(x7_0)
		x7 = self.bn7(x7)
		x7 = self.relu7(x7)
		x8_0 = x7 + x3
		x8 = self.conv8(x8_0)
		x8 = self.bn8(x8)
		x8 = self.relu8(x8)
		x9_0 = x8 + x2
		x9 = self.conv9(x9_0)
		x9 = self.bn9(x9)
		x9 = self.relu9(x9)
		x10_0 = x9 + x1
		x10 = self.conv10(x10_0)
		x10 = self.bn10(x10)
		x10 = self.relu10(x10)
		out = x10
		out = out + x
		
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
	model = RED_Residual_Net()
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

