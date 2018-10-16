#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by xuchongbo at 20171130 in Meitu.
"""

import argparse
import os
import shutil
import traceback
import time
import glob
from PIL import Image,ImageFilter
import scipy.misc
from scipy.misc import imresize
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#torch.backends.cudnn.enabled = False
import torchvision.transforms as transforms
import torchvision
import sys
import numpy as np
from squid import utils
from squid import myssim

debug = False


def process(input_tensor, target_net, gpu_id=None):
    start_time = time.time()
    input_tensor = input_tensor.view(1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))
    input_var = Variable(input_tensor, volatile=True)
    if debug:
        print "input sum:", input_var.sum(), input_var.size()
    if gpu_id is not None:
        input_var = input_var.cuda()
    input_var = input_var.cuda()
    end_time = time.time()
    if debug:
        print " load cost:%s ms" % int((end_time-start_time)*1000)
        print "inference.."
    start_time = time.time()
    # inference
    net_outputs = target_net(input_var)

    end_time = time.time()
    if debug:
        print " cost:%s ms" % int((end_time-start_time)*1000)
    return net_outputs


def run(input_dirs, save_dir, target_net, gpu_id=None, divided=True, psnr=False):
    if gpu_id is not None:
        print("use cuda")
        #cudnn.benchmark = True
        if gpu_id != -1:
            torch.cuda.set_device(gpu_id)
            target_net.cuda(gpu_id)
    print "input_dirs:", input_dirs
    print "save_dir:", save_dir
    utils.touch_dir(save_dir)
    if type(input_dirs) is not list:
        input_dirs = [input_dirs]  # 兼容旧config
    for input_dir in input_dirs: 
        # 样本目录
        if os.path.isdir(input_dir):
            files = utils.get_files_from_dir(input_dir)
        # 样本desc.txt
        elif os.path.isfile(input_dir):
            files = utils.get_files_from_desc(input_dir)
        total = 0
        for img_path in files:
            total += 1
            #read image
            img = Image.open(img_path)
            img = img.convert('RGB')
            #img = img.resize((384,384))
            width, height = img.size
            if debug:
                print width, height

            trans = transforms.Compose([transforms.ToTensor(), ])

            start_time = time.time()
            if (not divided) or (width < 1000) or (height < 1000):
                input_tensor = trans(img)
                net_outputs = process(input_tensor, target_net, gpu_id)

                end_time = time.time()
                if debug:
                    print "single image cost:%s ms" % int((end_time-start_time)*1000)

                for name, out in net_outputs.items():
                    save_path = os.path.join(save_dir, os.path.basename(img_path)+'_%s.png' % (name,))
                    print "save to :", save_path
                    utils.save_tensor(out.data[0], save_path, width, height)
            else:
                patch_size = 500
                crop_size = 300
                interval_size = (patch_size - crop_size) / 2
                input_tensor_0 = trans(img)

                output_1 = np.zeros((input_tensor_0.shape[0], input_tensor_0.shape[1], input_tensor_0.shape[2]))
                dict_1 = {'input': input_tensor_0, 'output': output_1}
                xx = 0
                while xx + patch_size < height:
                    yy = 0
                    while yy + patch_size < width:
                        input_tensor = trans(img.crop((yy, xx, yy + patch_size, xx + patch_size)))
                        net_outputs = process(input_tensor, target_net, gpu_id)
                        out_data = net_outputs['output'].data[0]
                        if xx == 0:
                            if yy == 0:
                                dict_1['output'][:, xx:xx+interval_size+crop_size, yy:yy+interval_size+crop_size] = out_data[:, 0:interval_size+crop_size, 0:interval_size+crop_size]
                            else:
                                dict_1['output'][:, xx:xx+interval_size+crop_size, yy+interval_size:yy+interval_size+crop_size] = out_data[:, 0:interval_size+crop_size, interval_size:interval_size+crop_size]
                        else:
                            if yy == 0:
                                dict_1['output'][:, xx+interval_size:xx+interval_size+crop_size, yy:yy+interval_size+crop_size] = out_data[:, interval_size:interval_size+crop_size, 0:interval_size+crop_size]
                            else:
                                dict_1['output'][:, xx+interval_size:xx+interval_size+crop_size, yy+interval_size:yy+interval_size+crop_size] = out_data[:, interval_size:interval_size+crop_size, interval_size:interval_size+crop_size]
                        yy += crop_size
                    yy = width - patch_size
                    input_tensor = trans(img.crop((yy, xx, yy + patch_size, xx + patch_size)))
                    net_outputs = process(input_tensor, target_net, gpu_id)
                    out_data = net_outputs['output'].data[0]
                    if xx == 0:
                        dict_1['output'][:, xx:xx+interval_size+crop_size, yy+interval_size:] = out_data[:, 0:interval_size+crop_size, interval_size:]
                    else:
                        dict_1['output'][:, xx+interval_size:xx+interval_size+crop_size, yy+interval_size:] = out_data[:, interval_size:interval_size+crop_size, interval_size:]
                    xx += crop_size
                xx = height - patch_size
                yy = 0
                while yy + patch_size < width:
                    input_tensor = trans(img.crop((yy, xx, yy + patch_size, xx + patch_size)))
                    net_outputs = process(input_tensor, target_net, gpu_id)
                    out_data = net_outputs['output'].data[0]
                    if yy == 0:
                        dict_1['output'][:, xx+interval_size:, yy:yy+interval_size+crop_size] = out_data[:, interval_size:, 0:interval_size+crop_size]
                    else:
                        dict_1['output'][:, xx+interval_size:, yy+interval_size:yy+interval_size+crop_size] = out_data[:, interval_size:, interval_size:interval_size+crop_size]
                    yy += crop_size
                yy = width - patch_size
                input_tensor = trans(img.crop((yy, xx, yy + patch_size, xx + patch_size)))
                net_outputs = process(input_tensor, target_net, gpu_id)
                out_data = net_outputs['output'].data[0]
                dict_1['output'][:, xx+interval_size:, yy+interval_size:] = out_data[:, interval_size:, interval_size:]

                end_time = time.time()
                if debug:
                    print "single image cost:%s ms" % int((end_time-start_time)*1000)

                for name, out in dict_1.items():
                    save_path = os.path.join(save_dir, os.path.basename(img_path)+'_%s.png' % (name,))
                    print "save to :", save_path
                    out = torch.Tensor(out).cuda()
                    utils.save_tensor(out, save_path, width, height)

        print "total: ", total

        if psnr == True:
            run_psnr(test_in_dir=input_dirs[0], test_out_dir=save_dir)

#SCALE = 8 
SCALE = 1

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def _open_img(img_p):
    F = scipy.misc.fromimage(Image.open(img_p)).astype(float)/255.0
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def _open_img_ssim(img_p):
    F = scipy.misc.fromimage(Image.open(img_p))#.astype(float)
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE 
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F


def run_psnr(test_in_dir, test_out_dir):
    print('run psnr and ssim ...')
    psnr_sum = 0
    ssim_sum = 0
    i_sum = 0
    if 'indoor' in test_in_dir:
        gt_dir='/root/group-competition/data/dehaze/NTIRE/indoor/test/gt'
        suffix_gt = '_indoor_GT.jpg'
    else:
        gt_dir='/root/group-competition/data/dehaze/NTIRE/outdoor/test/gt'
        suffix_gt = '_outdoor_GT.jpg'
    llist = os.listdir(test_out_dir)
    for line in llist:
        line.strip()
        lline = line
        suffix = line[-3:]
        if 'png' != suffix and 'jpg' != suffix:
            continue
        items = line.split('_')
        if items[3][0] == 'i':
            continue
        image1_dir = test_out_dir + '/' + lline
        image2_dir = gt_dir + '/' + items[0] + suffix_gt

        psnr = output_psnr_mse(_open_img(image1_dir), _open_img(image2_dir))
        psnr_sum += psnr

        image1 = _open_img_ssim(image1_dir)
        image2 = _open_img_ssim(image2_dir)
        channels = []
        for i in range(3):
            channels.append(myssim.compare_ssim(image1[:,:,i],image2[:,:,i], gaussian_weights=True, use_sample_covariance=False))
        ssim = np.mean(channels)
        ssim_sum += ssim

        i_sum += 1

    print('psnr:', psnr_sum / i_sum)
    print('ssim:', ssim_sum / i_sum)
    with open(test_out_dir + '/psnr.txt', 'w') as f:
        f.write('psnr: ' + str(psnr_sum / i_sum) + '\n')
        f.write('ssim: ' + str(ssim_sum / i_sum) + '\n')


if __name__ == "__main__":
    debug = True
    config = utils.load_config(sys.argv[1])
    # test_snapshot_path = sys.argv[2]
    checkpoint = torch.load(config.test_snapshot_path, map_location=lambda storage, loc: storage)
    print checkpoint.keys()
    config.target_net.load_state_dict(checkpoint)
    config.target_net.eval()
    run(config.test_input_dir, config.TEST_OUT_FOLDER, config.target_net, config.GPU_ID, config.divided, config.psnr)
