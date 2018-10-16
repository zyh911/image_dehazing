#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: xuchongbo at 20171130 in Meitu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class IouScore(nn.Module):
    def __init__(self, nclass):
        super(IouScore, self).__init__()
        self.nclass = nclass

    def _iou(self, a, b, label) :
        s = (a==label) + (b==label)
        ins = (s==2).sum()
        union = (s>=1).sum()
        return ins*1.0/union if union>0 else 0

    def forward(self, output, target):
        """
        output: B*1*H*W
        target: B*H*W
        """
        #softmax_output = F.softmax(output)
        #pred = softmax_output.data.max(1)[1]   # B*1*H*W
        #shape of target  B*H*W
        score_dict = {}
        for idx in range(self.nclass):
            #d[prefix+'_iou_'+name]  = self._iou(pred, target.data, idx)
            score_dict['class_%s' % idx] = self._iou(output.data, target.data, idx)
        return score_dict 
