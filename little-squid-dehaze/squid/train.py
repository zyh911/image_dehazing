#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created by xuchongbo at 20171130 in Meitu.
"""

import argparse
import os
import shutil
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch
import random
from PIL import Image
import sys
import os
from torch.autograd import Variable
from squid import utils
from squid import inference


config = utils.load_config(sys.argv[1])
# config = utils.load_config('./configs/xcb_20171031_srgan_dataset20171030_gauss_noise03_random_ds_random_kernel_jpeg_skip_1664_all_loss_L1_1_gan_5e2_v3.py')


def prepare_dirs():
    # assure 
    utils.touch_dir(config.MODEL_FOLDER)
    dirs = (("TrainOut", config.TRAIN_OUT_FOLDER), ("PeekOut",config.PEEK_OUT_FOLDER), ("TestOut",config.TEST_OUT_FOLDER))
    txt = ""
    for title, dirpath in dirs:
        utils.touch_dir(dirpath)
        # dirname = dirpath.rstrip("/").split('/')[-1]
        # name = config.experiment_name + "-" + dirname
        # utils.create_link(src=dirpath, dst=os.path.join(config.IMAGE_SITE_DATA_DIR,name))
        # url = config.IMAGE_SITE_URL.format(dataset_name=name)
        # txt += """ %s <a href='%s'> %s </a></br>""" % (title, url, url)

    # if config.vis is not None:
        # config.vis.text(txt, win='links', env=config.experiment_name)


def main():
    # raw_input("press anykey ")
    start_epoch = config.start_epoch  # epoch start from 1 
    if config.GPU_ID is not None:
        print("use cuda")
        #cudnn.benchmark = True
        torch.cuda.set_device(config.GPU_ID)
        config.model.cuda(config.GPU_ID)
    if config.only_validate:
        print("only validate it")
        valid_loader = torch.utils.data.DataLoader(dataset=config.valid_dataset, batch_size=config.batch_size,
                                                   shuffle=False, num_workers=2, drop_last=True)
        loss_dict = validate(valid_loader, config.model)
        print("validate:", loss_dict)
    else:
        # prepare dirs
        print ("prepare dirs and links")
        prepare_dirs()
        # Datasets
        print ("init data loader...")
        train_loader = torch.utils.data.DataLoader(dataset=config.train_dataset, batch_size=config.batch_size,
                                                   shuffle=True, num_workers=2,  pin_memory=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(dataset=config.valid_dataset, batch_size=config.validate_batch_size,
                                                   shuffle=False, num_workers=2,  pin_memory=True, drop_last=True)
        iters = len(train_loader)
        print("begin train..")
        for epoch in range(start_epoch, config.epochs+1):   
            train(epoch,  train_loader, config.model)
            if epoch % config.validate_interval_epoch == 0:
                loss_dict, score_dict = validate(valid_loader, config.model)
                utils.print_loss(config, "valid_loss", loss_dict, epoch, iters, iters, need_plot=True)
                utils.print_loss(config, "valid_score", score_dict, epoch, iters, iters, need_plot=True)
            if epoch % config.peek_interval_epoch == 0: 
                for item in config.peek_images:
                    peek(config.target_net, item, epoch)
            if epoch % config.save_snapshot_interval_epoch == 0:
                # Save the Models
                config.model.save_snapshot(os.path.join(config.MODEL_FOLDER, 'snapshot_%d' %(epoch)))

    # test model after train has completed. 
    inference.run(config.test_input_dir, config.TEST_OUT_FOLDER, config.target_net, config.GPU_ID)
    print "train and inference are completed."


def train(epoch, train_loader, model):
    loss_accumulator = utils.DictAccumulator(config.loss_average_win_size)
    grad_accumulator = utils.DictAccumulator(config.loss_average_win_size)
    score_accumulator = utils.DictAccumulator()
    iters = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)
        net_outputs, loss, grad, lr_dict, score = model.fit(inputs, targets, update=True, epoch=epoch)
        loss_accumulator.update(loss)
        grad_accumulator.update(grad)
        score_accumulator.update(score)
        if (i+1) % config.loss_average_win_size == 0:
            need_plot = True
            if hasattr(config, 'plot_loss_start_iter'):
                need_plot = (i + 1 + (epoch - 1) * iters >= config.plot_loss_start_iter)
            elif hasattr(config, 'plot_loss_start_epoch'):
                need_plot = (epoch >= config.plot_loss_start_epoch)

            utils.print_loss(config, "train_loss", loss_accumulator.get_average(), epoch=epoch, iters=iters, current_iter=i+1, need_plot=need_plot)
            utils.print_loss(config, "grad", grad_accumulator.get_average(), epoch=epoch, iters=iters, current_iter=i+1, need_plot=need_plot)
            utils.print_loss(config, "learning rate", lr_dict, epoch=epoch, iters=iters, current_iter=i+1, need_plot=need_plot)

            utils.print_loss(config, "train_score", score_accumulator.get_average(), epoch=epoch, iters=iters, current_iter=i+1, need_plot=need_plot)

    if epoch % config.save_train_hr_interval_epoch == 0:
        k = random.randint(0, net_outputs['output'].size(0) - 1)
        for name, out in net_outputs.items():
            utils.save_tensor(out.data[k], os.path.join(config.TRAIN_OUT_FOLDER, 'epoch_%d_k_%d_%s.png' % (epoch, k, name)))


def validate(valid_loader, model):
    loss_accumulator = utils.DictAccumulator()
    score_accumulator = utils.DictAccumulator()
    # loss of the whole validation dataset
    for i, (inputs, targets) in enumerate(valid_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets)
        loss, score = model.fit(inputs, targets, update=False)
        loss_accumulator.update(loss)
        score_accumulator.update(score)

    return loss_accumulator.get_average(), score_accumulator.get_average()


def peek(target_net, img_path, epoch):
    # open image
    img = Image.open(img_path)

    # save raw peek images for first time
    if epoch == config.peek_interval_epoch: 
        img.save(os.path.join(config.PEEK_OUT_FOLDER, os.path.basename(img_path)+'_0.png'))

    # do inference 
    img = img.convert('RGB')
    trans = transforms.Compose([transforms.ToTensor(), ])
    input_tensor = trans(img)
    inputs = input_tensor.view(1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))
    print("inference...")
    inputs = Variable(inputs, volatile=True)
    target_net.eval()
    net_outputs = target_net(inputs.cuda())
    # save net_outputs
    for name, out in net_outputs.items():
        utils.save_tensor(out.data[0], os.path.join(config.PEEK_OUT_FOLDER, os.path.basename(img_path)+'_%s_%d.png' % (name, epoch)))

        
if __name__ == '__main__':
    main()

