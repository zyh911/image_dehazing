#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Implementation of AOD-Net in ICCV2017
# Created by Zhao Yuhang in Meitu.

import os 
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class AOD_Deep1_Net(nn.Module):
	def __init__(self):
		super(AOD_Deep1_Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)
		self.relu3 = nn.ReLU(inplace=True)
		self.conv4 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)
		self.relu4 = nn.ReLU(inplace=True)
		self.conv5 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)
		self.relu5 = nn.ReLU(inplace=True)
		self.conv6 = nn.Conv2d(9, 3, kernel_size=7, stride=1, padding=3)
		self.relu6 = nn.ReLU(inplace=True)
		self.conv7 = nn.Conv2d(12, 3, kernel_size=7, stride=1, padding=3)
		self.relu7 = nn.ReLU(inplace=True)
		self.conv8 = nn.Conv2d(12, 3, kernel_size=5, stride=1, padding=2)
		self.relu8 = nn.ReLU(inplace=True)
		self.conv9 = nn.Conv2d(15, 3, kernel_size=5, stride=1, padding=2)
		self.relu9 = nn.ReLU(inplace=True)
		self.conv10 = nn.Conv2d(27, 3, kernel_size=3, stride=1, padding=1)
		self.relu10 = nn.ReLU()
		self.clip = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				#m.weight.data.normal_(0, 0.0001)

			elif isinstance(m, nn.ConvTranspose2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				#m.weight.data.normal_(0, 0.0001)

			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

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
		x5_0 = torch.cat((x3, x4), dim=1)
		x5 = self.conv5(x5_0)
		x5 = self.relu5(x5)
		x6_0 = torch.cat((x3, x4, x5), dim=1)
		x6 = self.conv6(x6_0)
		x6 = self.relu6(x6)
		x7_0 = torch.cat((x3, x4, x5, x6), dim=1)
		x7 = self.conv7(x7_0)
		x7 = self.relu7(x7)
		x8_0 = torch.cat((x4, x5, x6, x7), dim=1)
		x8 = self.conv8(x8_0)
		x8 = self.relu8(x8)
		x9_0 = torch.cat((x4, x5, x6, x7, x8), dim=1)
		x9 = self.conv9(x9_0)
		x9 = self.relu9(x9)
		x10_0 = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9), dim=1)
		x10 = self.conv10(x10_0)
		x10 = self.relu10(x10)
		out = x10 * x - x10 + 1
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
	model = AOD_Deep1_Net()
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

