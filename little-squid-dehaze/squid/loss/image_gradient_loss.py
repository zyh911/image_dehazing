import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np


class GradientHistLoss(nn.Module):
    """Gradient Histogram Loss"""
    def __init__(self):
        super(GradientHistLoss, self).__init__()
        self.bin_num = 64
        self.delta = 0.2
        self.clip_radius = 0.2 
        assert(self.clip_radius>0 and self.clip_radius<=1)
        self.bin_width = 2*self.clip_radius/self.bin_num
        if self.bin_width*255<1:
            raise RuntimeError("bin width is too small")
        self.bin_mean = np.arange(-self.clip_radius+self.bin_width*0.5, self.clip_radius, self.bin_width)
        self.gradient_hist_loss_function = 'L1'
        # default is KL loss
        if self.gradient_hist_loss_function == 'L2':
            self.criterion = nn.MSELoss()
        elif self.gradient_hist_loss_function == 'L1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.KLDivLoss()

    def get_response(self, gradient, mean):
        # tmp = torch.mul(torch.pow(torch.add(gradient, -mean), 2), self.delta_square_inverse)
        s = (-1) / (self.delta ** 2)
        tmp = ((gradient - mean) ** 2) * s
        return torch.mean(torch.exp(tmp))

    def get_gradient(self, src):
        right_src = src[:, :, 1:, 0:-1]     # shift src image right by one pixel
        down_src = src[:, :, 0:-1, 1:]      # shift src image down by one pixel
        clip_src = src[:, :, 0:-1, 0:-1]    # make src same size as shift version
        d_x = right_src - clip_src
        d_y = down_src - clip_src

        return d_x, d_y

    def get_gradient_hist(self, gradient_x, gradient_y):
        lx = None
        ly = None
        for ind_bin in range(self.bin_num):
            fx = self.get_response(gradient_x, self.bin_mean[ind_bin])
            fy = self.get_response(gradient_y, self.bin_mean[ind_bin])

            if lx is None:
                lx = fx
                ly = fy
            else:
                lx = torch.cat((lx, fx), 0)
                ly = torch.cat((ly, fy), 0)
        # lx = torch.div(lx, torch.sum(lx))
        # ly = torch.div(ly, torch.sum(ly))
        return lx, ly

    def forward(self, output, target):
        output_gradient_x, output_gradient_y = self.get_gradient(output)
        target_gradient_x, target_gradient_y = self.get_gradient(target)

        output_gradient_x_hist, output_gradient_y_hist = self.get_gradient_hist(output_gradient_x, output_gradient_y)
        target_gradient_x_hist, target_gradient_y_hist = self.get_gradient_hist(target_gradient_x, target_gradient_y)
        loss = self.criterion(output_gradient_x_hist, target_gradient_x_hist) + self.criterion(output_gradient_y_hist, target_gradient_y_hist)
        return loss


