#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class GanLoss(nn.Module):
    """
    GAN Loss
    TODO: use special optimizer for D, G in different gan settings. 
    """

    def __init__(self, args):
        super(GanLoss, self).__init__()
        assert(args['gan_setting'] in ['LSGAN', 'WGAN', 'DCGAN', 'Patch-GAN', 'WGAN-GP'])
        self.args = args
        self.bce_loss = nn.BCELoss()
        self.base_lr = self.args['lr']
        self.D = self.args['D']

        if self.args['gan_setting'] in ["WGAN",'WGAN-GP']:
            self.optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.base_lr)
            #self.G_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.G_lr)
        #elif 0:
            #self.optimizer = torch.optim.SGD(self.D.parameters(), lr=self.base_lr)
            #self.G_optimizer = torch.optim.SGD(self.G.parameters(), lr=self.G_lr)
        elif self.args['gan_setting'] in ["DCGAN","Patch-GAN"]:
            #self.optimizer = torch.optim.Adam(D.parameters(), lr=10e-5,  betas=(0.5, 0.999))
            #best self.optimizer = torch.optim.RMSprop(self.D.parameters(), lr=5e-5)
            self.optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.base_lr)
            #self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.G_lr)
    
    def partial_loss(self, d_output, required_real):
        if self.args['gan_setting'] in ["WGAN", 'WGAN-GP']:
            if required_real:
                loss = -torch.mean(d_output) 
            else:
                loss = torch.mean(d_output)
        elif self.args['gan_setting'] == "LSGAN":
            if required_real:
                loss = torch.mean((d_output - 1)**2) 
            else:
                loss = torch.mean(d_output**2)
        elif self.args['gan_setting'] == "DCGAN":
            if required_real:
                real_labels = Variable(torch.ones(d_output.size(0)).cuda())
                loss = self.bce_loss(d_output, real_labels)
            else:
                fake_labels = Variable(torch.zeros(d_output.size(0)).cuda())
                loss = self.bce_loss(d_output, fake_labels)
        elif self.args['gan_setting'] == 'Patch-GAN':
            if required_real:
                real_labels = Variable(torch.ones(d_output.size()).cuda())
                loss = self.bce_loss(d_output, real_labels)
            else:
                fake_labels = Variable(torch.zeros(d_output.size()).cuda())
                loss = self.bce_loss(d_output, fake_labels)
        return loss


    def forward(self, output, target): 
        #output_vgg = self.vgg_feature(output)
        #target_vgg = self.vgg_feature(target.detach())
        self.D.eval()
        real_d_loss = self.partial_loss(self.D(output), required_real=True)
        return real_d_loss 

    def fit(self, output, target, epoch): 
            self._adjust_learning_rate(epoch)
            self.D.train()
            self.D.zero_grad()
            real_d_output = self.D(Variable(target.data.clone(), requires_grad=True))
            fake_d_output = self.D(Variable(output.data.clone(), requires_grad=True)) 
            real_d_loss = self.partial_loss(real_d_output, required_real=True)
            fake_d_loss = self.partial_loss(fake_d_output, required_real=False)
            d_loss = (real_d_loss + fake_d_loss) * 0.5
            d_loss.backward()
            # train with gradient penalty
            #gradient_penalty = self.calc_gradient_penalty(hr_imgs.data, sr_imgs.data)
            #gradient_penalty.backward()
            self.optimizer.step()
            if self.args['gan_setting'] == "WGAN":
                for p in self.D.parameters():
                    p.data.clamp_(-0.01, 0.01)
            # for plot
            return {'sum_d_loss':d_loss.data[0], 'real_d_loss':real_d_loss.data[0], 'fake_d_loss': fake_d_loss.data[0]}

    def _adjust_learning_rate(self, epoch):
        for param_group in self.optimizer.param_groups:
                lr_step_ratio = self.args['lr_step_ratio']
                lr_step_size = self.args['lr_step_size']
                param_group['lr'] = self.base_lr * np.power(lr_step_ratio, np.int(epoch / lr_step_size))

if __name__ == "__main__":
    pass
