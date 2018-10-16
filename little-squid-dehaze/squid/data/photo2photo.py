#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageFilter
import random
import io
import os
import os.path
import numpy as np
import numbers

class Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.image_list = self.read(self.args['desc_file_path'])
        # Image Preprocessing
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        input_path, target_path = self.image_list[index]
        input_img = Image.open(input_path)  # h, w, c
        target_img = Image.open(target_path)  # h, w, c
        # CxHxW
        input_tensor = torch.from_numpy(np.asarray(input_img).transpose((2, 0, 1)))
        input_tensor = input_tensor.float().div(255)

        # CxHxW
        target_tensor = torch.from_numpy(np.asarray(target_img).transpose((2, 0, 1)))
        target_tensor = target_tensor.float().div(255)

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.image_list)


    def read(self, desc_file_path):
        import codecs
        result = []
        data_root = self.args.get("data_root", None)
        with codecs.open(desc_file_path, 'r', 'utf-8') as txt:
            for line in txt:
                line = line.strip()
                input_path, target_path = line.encode('utf-8').split(' ')
                if data_root:
                    input_path  = os.path.join(data_root, input_path)
                    target_path = os.path.join(data_root, target_path)
                result.append((input_path, target_path))
            print "read from {} total {} ".format(desc_file_path, len(result))
        return result

