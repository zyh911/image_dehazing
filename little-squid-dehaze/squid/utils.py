#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path
import torch
import sys
import socket
import time
import traceback
from urlparse import urlparse
import cStringIO
import PIL.Image
from PIL import Image
import numpy as np
from collections import OrderedDict


all_wins = {}


def clear_wins(config):
    if config.vis is not None:
        config.vis.close(win=None, env=config.experiment_name)

"""
logging.basicConfig(
     stream=sys.stdout,
     level=logging.DEBUG,
     format='[%(asctime)s] [%(name)s] [%(levelname)s]:\t%(message)s')
logger = logging.getLogger('mylog')
"""


def print_loss(config, title, loss_dict, epoch, iters, current_iter, need_plot=False):
    data_str = ''
    for k, v in loss_dict.items():
        if data_str != '':
            data_str += ', '
        data_str += '{}: {:.10f}'.format(k, v)

        if need_plot and config.vis is not None:
            plot_line(config, title, k, (epoch-1)*iters+current_iter, v)

    # step is the progress rate of the whole dataset (split by batchsize)
    print('[{}] [{}] Epoch [{}/{}], Iter [{}/{}]'.format(title, config.experiment_name, epoch, config.epochs, current_iter, iters))
    print('        {}'.format(data_str))


def plot_line(config, title, name, i, v):
    win = all_wins.get(title, None)
    if win is None:
        win = config.vis.line(env=config.experiment_name, X=np.array([i]), Y=np.array([v]), opts={'legend':[name], 'title':title})
        all_wins[title] = win
    else:
        config.vis.updateTrace(env=config.experiment_name, win=win,  X=np.array([i]), Y=np.array([v]), name=name)


def plot_IOU(config, win, title, acc_dict, best_dict):
    """
    viz.bar( X=np.random.rand(20, 3), opts=dict( stacked=False, legend=['The Netherlands', 'France', 'United States']))
    """
    a = np.array(acc_dict.values())
    b = np.array(best_dict.values())
    print a.shape, b.shape
    data = np.vstack((a,b))
    #data = data.reshape((data.shape[0],1))
    data = data.transpose((1, 0))
    print data.shape
    legend = [config.experiment_name, "best"]
    if config.vis is not None:
        config.vis.bar(env=config.experiment_name, win=win, X=data, Y=best_dict.keys(),
                       opts=dict(stacked=False, legend=legend, title=title))


def print_model(model):
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)


def touch_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(src, dst)


def get_network_ip():
    return "172.18.11.163"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0))
    return s.getsockname()[0]

def to_palette_img(img):
    from collections import OrderedDict
    COLOER_MAP= OrderedDict( (('BACK_GROUND', (0,0,0)),
                    ('FACE_SKIN', (204,202,169)),
                    ('LEFT_BROW', (255,165,0)),
                    ('RIGHT_BROW', (255,0,255)),
                    ('LEFT_EYE',  (0,255,0)),
                    ('RIGHT_EYE', (0,0,255)),
                    ('NOSE' , (0, 255, 255)),
                    ('UPPER_LIP', (255,255,0)),
                    ('LOWER_LIP',(255,0,0)),
                    ('TEETH' , (255,255,255)),
                    ('LEFT_EAR',(0,100,0)),
                    ('RIGHT_EAR' , (106,90,205)),
                    ('LEFT_PUPILLA', (100,149,237)),
                    ('RIGHT_PUPILLA', (139,101,139)),
                    ('GLASSES', (172,53,79)),
                    ('BEARD',  (167,167,177)) ))


    # create a lookup table (r, g, b, r, g, b, r, g, b, ...)
    lut = []
    colors = COLOER_MAP.values()
    for color in colors:
        #lut.extend([255-i, i/2, i])
        lut.extend(color)
        img.putpalette(lut)
        assert img.mode == "P" # now has a palette
    return img

def save_tensor(input_tensor, save_path, width=None, height=None):
    """
    input_tensor is in (0, 1)
    """
    input_tensor = input_tensor.cpu()
    if type(input_tensor) not in [torch.IntTensor, torch.LongTensor]:
        input_tensor = input_tensor.mul(255)
    ndarr= input_tensor.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    ndarr = np.squeeze(ndarr)
    out_img = Image.fromarray(ndarr)
    if type(input_tensor) in [torch.IntTensor, torch.LongTensor]:
        out_img = to_palette_img(out_img)
    w, h = out_img.size
    if width is not None and (w, h) != (width, height):
        out_img = out_img.resize((width, height), Image.BICUBIC)
    # prepare dir
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
	    os.makedirs(save_dir)
    # save
    out_img.save(save_path)


def tensor_to_pil(input_tensor):
    np_img = input_tensor.byte().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    img = Image.fromarray(np_img)

    return img


def pil_to_tensor(pic):
    # handle PIL Image
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    img = img.view(pic.size[1], pic.size[0], len(pic.mode))
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).float()

    return img


def save_image(path, img):
    data_dir = os.path.dirname(path)
    if not os.path.exists(data_dir):
	    os.makedirs(data_dir)
    cv2.imwrite(path, img)


def to_image_ndarr(input_tensor):
    ndarr= input_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return ndarr


def merge_tow_numpy_and_save_to_image(img1, img2, filepath):
    alpha = 0.55
    out = np.asarray(img1*alpha + (1-alpha)*img2, dtype=np.uint8)
    im = Image.fromarray(out)
    im.save(filepath)


def pad_image(im, out_h, out_w):
    assert(im.width <= out_w)
    assert(im.height <= out_h)
    out = Image.new('RGB', (out_w, out_h), (0, 0, 0)) 
    out.paste(im, (0, 0))
    return out


def is_url(url):
    return url is not None and urlparse(url).scheme != "" and not os.path.exists(url)


def get_cvmat(path):    
    import requests
    if is_url(path):
            r = requests.get(path,
                    allow_redirects=False,
                    timeout=2)
            r.raise_for_status()
            stream = cStringIO.StringIO(r.content)
            image = PIL.Image.open(stream)
    else:
        image = PIL.Image.open(path)
    imgcv = np.asarray(image.convert('L'))
    return imgcv


class AverageWithinWindow():
    def __init__(self, win_size):
        self.win_size = win_size
        self.cache = []
        self.average = 0
        self.count = 0

    def update(self, v):
        if self.count < self.win_size:
            self.cache.append(v)
            self.count += 1
            self.average = (self.average * (self.count - 1) + v) / self.count
        else:
            idx = self.count % self.win_size
            self.average += (v - self.cache[idx]) / self.win_size
            self.cache[idx] = v
            self.count += 1


class DictAccumulator():
    def __init__(self, win_size=None):
        self.accumulator = OrderedDict()
        self.total_num = 0 
        self.win_size = win_size

    def update(self, d):
        self.total_num += 1
        for k, v in d.items():
            if not self.win_size:
                self.accumulator[k] = v + self.accumulator.get(k,0)
            else:
                self.accumulator.setdefault(k, AverageWithinWindow(self.win_size)).update(v)

    def get_average(self):
        average = OrderedDict()
        for k, v in self.accumulator.items():
            if not self.win_size:
                average[k] = v*1.0/self.total_num 
            else:
                average[k] = v.average 
        return average


def printGPUINFO():
    gpu_id = config.GPU_ID
    gpu_obj = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    print ("gup mem used:", pynvml.nvmlDeviceGetMemoryInfo(gpu_obj).used/1024/1024, "MB")


def memory_used(model, images):
    print ("before")
    printGPUINFO()
    model.cuda(config.GPU_ID)
    print ("do forward...")
    outputs = model(images.cuda(config.GPU_ID))
    print type(outputs.data)
    print ("after")
    printGPUINFO()
    print (outputs.size())   # (10, 100)


def load_config(file_path):
    file_path = os.path.abspath(file_path)
    print "load config file:", file_path

    sys.path.append(os.path.dirname(file_path))
    dir_path, _, file_py = file_path.rpartition('/')
    file_name = file_py[:-3]
    config = __import__(file_name)
    d = config.__dict__
    ret = {}
    txt = ""
    for k in d:
        if not k.startswith("__") and not callable(d[k]):
            ret[k] = d[k]
            txt += "%s = %s </br>" % (k, d[k])
            print k, ':', d[k]

    if config.vis is not None:
        config.vis.text(txt, win='args', env=config.experiment_name)

    return config


def get_files_from_desc(file_path):
    import codecs
    with codecs.open(file_path, "r", 'utf-8') as my_file:
        i = 0
        for line in my_file:
            i += 1
            image_path = line.strip().encode('utf-8')
            try:
                yield image_path
            except Exception, e:
                print "Exception happens:", e
                print traceback.format_exc()
                time.sleep(2)


def get_files_from_dir(root_dir, exts=['.jpg', '.png', '.jpeg', '.gif', '.BMP', '.bmp']):
    root_dir = root_dir.rstrip("/")
    i = 0
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename[-4:] in exts:
                i += 1
                image_path = os.path.join(root, filename)
                try:
                    yield image_path
                except Exception, e:
                    print "Exception happens:", e
                    print traceback.format_exc()
                    time.sleep(2)

