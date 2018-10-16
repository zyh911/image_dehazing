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

class MyRandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, padding=0, stride=1):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.stride = stride

    def __call__(self, lr_img, hr_img):
        assert(lr_img.size==hr_img.size)
        if self.padding > 0:
            lr_img = ImageOps.expand(lr_img, border=self.padding, fill=0)
            hr_img = ImageOps.expand(hr_img, border=self.padding, fill=0)

        w, h = lr_img.size
        th, tw = self.size
        if w == tw and h == th:
            return lr_img, hr_img

        #x1 = random.randint(0, w - tw)  #[0, w-tw]
        #y1 = random.randint(0, h - th)
        x1 = random.randrange(0, w-tw+1, self.stride)  #[0, w-tw+1)
        y1 = random.randrange(0, h-th+1, self.stride)
        return lr_img.crop((x1, y1, x1 + tw, y1 + th)), hr_img.crop((x1, y1, x1 + tw, y1 + th))

class Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.image_list = self.read(self.args['desc_file_path'])
        # Image Preprocessing
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, index):
        lr_path, hr_path = self.image_list[index]
        lr_img = Image.open(lr_path)  # h, w, c
        hr_img = Image.open(hr_path)  # h, w, c
        #hr_img.show()
        if self.args.get('crop_size', 0) > 0:
            self.randomCrop = MyRandomCrop(self.args['crop_size'], stride=self.args['crop_stride'])
            lr_img, hr_img = self.randomCrop(lr_img, hr_img)
        return self.toTensor(lr_img), self.toTensor(hr_img) 

    def __len__(self):
        return len(self.image_list)


    def read(self, desc_file_path):
        import codecs
        result = []
        with codecs.open(desc_file_path, 'r', 'utf-8') as txt:
            for line in txt:
                line = line.strip()
                lr_path, hr_path = line.encode('utf-8').split(' ')
                lr_path = os.path.join(self.args['data_root'], lr_path)
                hr_path = os.path.join(self.args['data_root'], hr_path)
                result.append((lr_path, hr_path))
            print "read from {} total {} ".format(desc_file_path, len(result))
        return result

