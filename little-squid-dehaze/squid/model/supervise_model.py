#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by xuchongbo at 20171130 in Meitu.
"""


from __future__ import print_function
import torch
import math
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np


class SuperviseModel(nn.Module):
    def __init__(self, args):
        super(SuperviseModel, self).__init__()
        self.args = args
        self.optimizer = self.args['optimizer']
        # create model
        self.net = self.args['net']
        print('net type:', type(self.net))

        # record loss instances
        self.loss_instances  = set()
        for out_name, required_loss in self.args['supervise'].items():
            if required_loss is None:
                continue
            for loss_name, loss in required_loss.items():
                #loss_inst.name = out_name+'_'+loss_name
                loss_inst = loss['obj']
                self.loss_instances.add(loss_inst)
                setattr(self, out_name+'_'+loss_name, loss_inst)  # for cuda() to send loss to gpu

    def _compute_loss(self, pairs, need_backward=False, epoch=None):
        plot_loss_dict = {}
        plot_grad_dict = {}
        for out_name, out_losses in self.args['supervise'].items():
            output, target = pairs[out_name]
            loss_dict = {}
            weight_dict = {}
            output_copy = Variable(output.data.clone(), requires_grad=True)
            target = target.detach()
            # compute all loss for this output
            for k, item in out_losses.items():
                if item['obj']:
                    # compute loss
                    loss = item['obj'](output_copy, target)*item['factor']
                    # cahe it
                    kk = out_name+'_'+k
                    loss_dict[kk] = loss
                    weight_dict[kk] = item['weight']

                    # train learnable loss
                    if need_backward and getattr(item['obj'], 'fit', None):
                        plot_loss_fit_dict = item['obj'].fit(output_copy, target, epoch)
                        plot_loss_dict.update({kk+'_'+key:plot_loss_fit_dict[key] for key in plot_loss_fit_dict})

            if need_backward:
                # grad
                grad_sum, ret_grad_dict = self._grad_wrt_output(output_copy, loss_dict, weight_dict)
                if len(ret_grad_dict) > 1:
                    ret_grad_dict[out_name+'_grad_sum'] = grad_sum.mean()

                plot_grad_dict.update(ret_grad_dict)
                output.backward(grad_sum, retain_variables=True) # set retain_variables in case for share network in low layers

            if len(loss_dict) > 1:
                loss_dict[out_name+'_loss_sum'] = sum([loss_dict[k] for k in loss_dict])

            plot_loss_dict.update({k:loss_dict[k].data[0] for k in loss_dict})

        if need_backward:
            return plot_loss_dict, plot_grad_dict
        else:
            return plot_loss_dict 


    def _compute_score(self, pairs):
        plot_score_dict = {}
        for out_name, out_metrics in self.args['metrics'].items():
            output, target = pairs[out_name]
            output_copy = Variable(output.data.clone(), requires_grad=True)
            target = target.detach()
            for k, item in out_metrics.items():
                    score = item['obj'](output_copy, target)
                    kk = out_name + '_' + k
                    if type(score) is dict:
                        plot_score_dict.update({kk+'_'+key:score[key] for key in score})
                    else:
                        plot_score_dict[kk] = score
        return plot_score_dict 

    def save_snapshot(self, path):
        torch.save(self.net.state_dict(), path+'_G_model')

    def _adjust_learning_rate(self, epoch):
        # adjust lr for net's optimizer
        for param_group in self.optimizer.param_groups:
                lr_step_ratio = self.args['lr_step_ratio']
                lr_step_size = self.args['lr_step_size']
                param_group['lr'] = param_group['base_lr'] * np.power(lr_step_ratio, np.int(epoch / lr_step_size))

    def _grad_wrt_output(self, output_copy, loss_dict, weight_dict):
        grad_sum = 0
        plot_grad_dict = {}
        for k in loss_dict:
            loss_dict[k].backward()
            grad = output_copy.grad.data
            grad_sum += grad*weight_dict[k]
            plot_grad_dict[k+'_grad'] = grad.mean()
            #print("{0:30}:  {1:15.10f}".format(kk, plot_grad_dict[kk]))
            output_copy.grad.data.zero_()
        return grad_sum, plot_grad_dict 

    def fit(self, lr_imgs, hr_imgs, update=True, epoch=None):
        if not update:
            self.net.eval()
            pairs, _ = self.net(lr_imgs, target=hr_imgs)
            plot_loss_dict = self._compute_loss(pairs)
            plot_score_dict = self._compute_score(pairs)
            return plot_loss_dict, plot_score_dict
        else:
            self._adjust_learning_rate(epoch)

            self.net.train()
            # forward
            pairs, net_outputs = self.net(lr_imgs, target=hr_imgs)
            # compute loss and do backward
            self.net.zero_grad()
            plot_loss_dict, plot_grad_dict = self._compute_loss(pairs, need_backward=True, epoch=epoch)
            plot_score_dict = self._compute_score(pairs)
            self.optimizer.step()
            # for plot
            lr_dict = {}
            for param_group in self.optimizer.param_groups:
                lr_dict[param_group['name']+'_lr'] = param_group['lr']
            return net_outputs, plot_loss_dict, plot_grad_dict, lr_dict, plot_score_dict 

