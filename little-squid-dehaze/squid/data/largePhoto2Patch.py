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
import PIL

class Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.image_list = self.read(self.args['desc_file_path'])
        self.input_dict, self.target_dict = self.read_img_to_tensor()

        self.patch_size = self.args['patch_size']
        self.is_train = self.args['is_train']
        if not self.is_train:
            self.patch_stride = self.args['patch_stride']
            self.patch_location_list, self.patch_num = self.generate_patch_location()
        else:
            self.patch_num = self.args['patch_num']


        # Image Preprocessing
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        if self.is_train:
            if self.args.get('var_threshold', None) is not None:
                var_threshold = self.args['var_threshold']
                while(True):
                    img_index = np.random.randint(0, len(self.image_list))
                    c, h, w = self.input_dict[img_index].size()
                    crop_h = np.random.randint(0, h - self.patch_size + 1)
                    crop_w = np.random.randint(0, w - self.patch_size + 1)
                    input_tensor = self.input_dict[img_index][:, crop_h:crop_h + self.patch_size,
                                   crop_w:crop_w + self.patch_size]
                    target_tensor = self.target_dict[img_index][:, crop_h:crop_h + self.patch_size,
                                    crop_w:crop_w + self.patch_size]
                    input_tensor_contiguous = input_tensor.contiguous()
                    input_tensor_view = input_tensor_contiguous.view(input_tensor_contiguous.size()[0], -1)
                    input_tensor_var = torch.mean(torch.std(input_tensor_view, dim = 1 , unbiased = False))
                    if input_tensor_var <= var_threshold:
                        continue
                    else:
                        break


            else:
                img_index = np.random.randint(0, len(self.image_list))
                c, h, w = self.input_dict[img_index].size()
                crop_h = np.random.randint(0, h - self.patch_size + 1)
                crop_w = np.random.randint(0, w - self.patch_size + 1)
                input_tensor = self.input_dict[img_index][:, crop_h:crop_h + self.patch_size,
                               crop_w:crop_w + self.patch_size]
                target_tensor = self.target_dict[img_index][:, crop_h:crop_h + self.patch_size,
                                crop_w:crop_w + self.patch_size]




        else:
            img_index, crop_h, crop_w = self.patch_location_list[index]
            input_tensor = self.input_dict[img_index][:,crop_h:crop_h + self.patch_size, crop_w:crop_w + self.patch_size]
            target_tensor = self.target_dict[img_index][:, crop_h:crop_h + self.patch_size, crop_w:crop_w + self.patch_size]

        return input_tensor, target_tensor

    def __len__(self):
        return self.patch_num


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

    def read_img_to_tensor(self):

        img_num = len(self.image_list)
        input_dict = {}
        target_dict = {}




        flip_ops = [PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.FLIP_TOP_BOTTOM, PIL.Image.ROTATE_90,
                    PIL.Image.ROTATE_180, PIL.Image.ROTATE_270, 'no_flip']
        #flip_ops = [PIL.Image.FLIP_LEFT_RIGHT]

        count = 0

        for i in range(img_num):
            print 'img ', i
            for flip_op in flip_ops:

                input_path, target_path = self.image_list[i]
                input_img = Image.open(input_path)  # h, w, c
                target_img = Image.open(target_path)  # h, w, c
                if flip_op is not 'no_flip':
                    input_img = input_img.transpose(method = flip_op)
                    target_img = target_img.transpose(method = flip_op)



                if self.args.get('awb', None) is not None:
                    r_gain, g_gain, b_gain = self.cal_awb_param(input_img)



                input_tensor = torch.from_numpy(np.asarray(input_img).transpose((2, 0, 1)))
                input_tensor = input_tensor.float().div(255)
                if self.args.get('awb', None) is not None:
                    input_tensor[0,:,:] = input_tensor[0,:,:] * r_gain
                    input_tensor[1,:,:] = input_tensor[1,:,:] * g_gain
                    input_tensor[2,:,:] = input_tensor[2,:,:] * b_gain

                # CxHxW
                target_tensor = torch.from_numpy(np.asarray(target_img).transpose((2, 0, 1)))
                target_tensor = target_tensor.float().div(255)
                if self.args.get('awb', None) is not None:
                    target_tensor[0,:,:] = target_tensor[0,:,:] * r_gain
                    target_tensor[1,:,:] = target_tensor[1,:,:] * g_gain
                    target_tensor[2,:,:] = target_tensor[2,:,:] * b_gain



                input_dict[count] = input_tensor
                target_dict[count] = target_tensor
                count = count + 1

        return input_dict, target_dict

    def generate_patch_location(self):
        img_num = len(self.image_list)
        patch_location_list = []
        for i in range(img_num):
            c, h, w = self.input_dict[i].size()
            for j in range(0, h - self.patch_size + 1, self.patch_stride) + [h - self.patch_size]:
                for k in range(0, w - self.patch_size + 1, self.patch_stride) + [w - self.patch_size]:
                    patch_location_list.append([i, j, k])


        return patch_location_list, len(patch_location_list)


    def cal_awb_param(self, img):
        img_array = np.asarray(img)
        img_array = img_array.transpose(2,0,1)
        r_max = img_array[0].max()
        g_max = img_array[1].max()
        b_max = img_array[2].max()

        return 255.0/r_max, 255.0/g_max, 255.0/b_max











