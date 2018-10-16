#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Implementation of AOD-Net in ICCV2017
# Created by Zhao Yuhang in Meitu.

import os 
import torch
import torch.nn as nn
from torch.autograd import Variable

class Unet_AOD_Deep1_Net(nn.Module):
	def __init__(self):
		super(Unet_AOD_Deep1_Net, self).__init__()
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
		self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU(inplace=True)
		self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU(inplace=True)
		self.conv8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn8 = nn.BatchNorm2d(32)
		self.relu8 = nn.ReLU(inplace=True)
		self.conv9 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn9 = nn.BatchNorm2d(32)
		self.relu9 = nn.ReLU(inplace=True)
		self.conv10 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn10 = nn.BatchNorm2d(32)
		self.relu10 = nn.ReLU(inplace=True)

		self.conv11 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.bn11 = nn.BatchNorm2d(32)
		self.relu11 = nn.ReLU(inplace=True)
		self.conv12 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.bn12 = nn.BatchNorm2d(32)
		self.relu12 = nn.ReLU(inplace=True)
		self.conv13 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.bn13 = nn.BatchNorm2d(32)
		self.relu13 = nn.ReLU(inplace=True)
		self.conv14 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.bn14 = nn.BatchNorm2d(32)
		self.relu14 = nn.ReLU(inplace=True)
		self.conv15 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.bn15 = nn.BatchNorm2d(32)
		self.relu15 = nn.ReLU(inplace=True)
		self.conv16 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.bn16 = nn.BatchNorm2d(32)
		self.relu16 = nn.ReLU(inplace=True)
		self.conv17 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.bn17 = nn.BatchNorm2d(32)
		self.relu17 = nn.ReLU(inplace=True)
		self.conv18 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.bn18 = nn.BatchNorm2d(32)
		self.relu18 = nn.ReLU(inplace=True)
		self.conv19 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.bn19 = nn.BatchNorm2d(32)
		self.relu19 = nn.ReLU(inplace=True)
		self.conv20 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
		self.bn20 = nn.BatchNorm2d(3)
		self.relu20 = nn.ReLU(inplace=True)

		self.clip = nn.ReLU(inplace=True)

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
		x7_0 = x6
		x7 = self.conv7(x7_0)
		x7 = self.bn7(x7)
		x7 = self.relu7(x7)
		x8_0 = x7
		x8 = self.conv8(x8_0)
		x8 = self.bn8(x8)
		x8 = self.relu8(x8)
		x9_0 = x8
		x9 = self.conv9(x9_0)
		x9 = self.bn9(x9)
		x9 = self.relu9(x9)
		x10_0 = x9
		x10 = self.conv10(x10_0)
		x10 = self.bn10(x10)
		x10 = self.relu10(x10)
		x11_0 = x10
		x11 = self.conv11(x11_0)
		x11 = self.bn11(x11)
		x11 = self.relu11(x11)

		x12_0 = torch.cat((x9, x11), dim=1)
		x12 = self.conv12(x12_0)
		x12 = self.bn12(x12)
		x12 = self.relu12(x12)
		x13_0 = torch.cat((x8, x12), dim=1)
		x13 = self.conv13(x13_0)
		x13 = self.bn13(x13)
		x13 = self.relu13(x13)
		x14_0 = torch.cat((x7, x13), dim=1)
		x14 = self.conv14(x14_0)
		x14 = self.bn14(x14)
		x14 = self.relu14(x14)
		x15_0 = torch.cat((x6, x14), dim=1)
		x15 = self.conv15(x15_0)
		x15 = self.bn15(x15)
		x15 = self.relu15(x15)
		x16_0 = torch.cat((x5, x15), dim=1)
		x16 = self.conv16(x16_0)
		x16 = self.bn16(x16)
		x16 = self.relu16(x16)
		x17_0 = torch.cat((x4, x16), dim=1)
		x17 = self.conv17(x17_0)
		x17 = self.bn17(x17)
		x17 = self.relu17(x17)
		x18_0 = torch.cat((x3, x17), dim=1)
		x18 = self.conv18(x18_0)
		x18 = self.bn18(x18)
		x18 = self.relu18(x18)
		x19_0 = torch.cat((x2, x18), dim=1)
		x19 = self.conv19(x19_0)
		x19 = self.bn19(x19)
		x19 = self.relu19(x19)
		x20_0 = torch.cat((x1, x19), dim=1)
		x20 = self.conv20(x20_0)
		x20 = self.bn20(x20)
		x20 = self.relu20(x20)
		

		out = x20 * x - x20 + 1
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
	model = Unet_AOD_Deep1_Net()
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

