#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by xuchongbo at 20171130 in Meitu.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class AccScore(nn.Module):
    def __init__(self):
        super(AccScore, self).__init__()

    def forward(self, output, target):
        """
        output: B*1*H*W
        target: B*H*W
        """
        #softmax_output = F.softmax(output)
        #pred = softmax_output.data.max(1)[1]   # B*1*H*W
        #shape of target  B*H*W
        score = (output.data == target.data).sum() / float(target.data.size(0) * target.data.size(1) * target.data.size(2))
        return score

