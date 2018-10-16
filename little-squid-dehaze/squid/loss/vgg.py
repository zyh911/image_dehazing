#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable


def make_layers(batch_norm=False):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG19FeatureTF(nn.Module):  # TensorFlow VGG19
    def __init__(self, vgg19_feature_model_path='', out_feature_level=4):
        super(VGG19FeatureTF, self).__init__()
        self.features = make_layers()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [1.0, 1.0, 1.0]

        self.load_state_dict(torch.load(vgg19_feature_model_path))
        if 1 < out_feature_level < 5:
            layer_ind = 37 - (5 - out_feature_level) * 9
            self.features = nn.Sequential(*list(self.features.children())[:layer_ind])
        elif out_feature_level == 1:
            self.features = nn.Sequential(*list(self.features.children())[:5])

        # print self.features

    def _normalize(self, x):
        out = torch.stack([(x[:, 0, :, :] - self.mean[0]) / self.std[0],
                           (x[:, 1, :, :] - self.mean[1]) / self.std[1],
                           (x[:, 2, :, :] - self.mean[2]) / self.std[2]], dim=1)

        out = out * 255.0

        return out

    def forward(self, x):
        x = self._normalize(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)

        return x


class VGGLoss(nn.Module):
    """Gradient Histogram Loss"""
    def __init__(self, vgg19_feature_model_path='', out_feature_level=4):
        super(VGGLoss, self).__init__()
        self.vgg_feature = VGG19FeatureTF(vgg19_feature_model_path=vgg19_feature_model_path,
                                          out_feature_level=out_feature_level)
        self.distance_type = 'L2'
        if self.distance_type == 'L2':
            self.criterion = nn.MSELoss()
        elif self.distance_type == 'L1':
            self.criterion = nn.L1Loss()

    def forward(self, output, target):  # if taget vailite depends on user
        output_vgg = self.vgg_feature(output)
        target_vgg = self.vgg_feature(target.detach())
        loss = self.criterion(output_vgg, target_vgg.detach())
        return loss


if __name__ == "__main__":
    torch.cuda.set_device(0)
    images = Variable(torch.ones(1, 3, 128, 128)).cuda()
    vgg = VGG19FeatureTF()
    vgg.cuda()
    print "do forward..."
    outputs = vgg(images)
    print (outputs.size())   # (10, 100)
    print torch.max(outputs)
