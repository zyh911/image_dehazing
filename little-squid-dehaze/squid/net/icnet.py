import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class init_s2(nn.Module):
    def __init__(self, inplanes, planes, outplanes):
        super(init_s2, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, outplanes, kernel_size=3, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.convs(x)
        return out

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, outplanes, stride=1, pad = 1, dilation = 1):
        super(Bottleneck, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, planes, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(),

            nn.Conv2d(planes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
        )
        self.inplanes  = inplanes
        self.outplanes = outplanes
        if inplanes != outplanes:
            self.convm = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.convs(x)
        if self.inplanes != self.outplanes:
            residual = self.convm(residual)

        out += residual
        out = self.relu(out)
        return out

class psp_pooling(nn.Module):
    def __init__(self, scale, stride, output_size):
        super(psp_pooling, self).__init__()
        self.scale = scale
        self.stride = stride
        self.output_size = output_size
        #self.res = [None]*len(self.scale)


    def forward(self, x):
        '''out = x
        for i in xrange(len(self.scale)):
            self.res[i] = F.avg_pool2d(x, self.scale[i], self.stride[i])
            self.res[i] = F.upsample_bilinear(self.res[i], size=self.output_size)
            out += self.res[i]'''
        res1 = F.avg_pool2d(x, self.scale[0], self.stride[0])
        res1_up = F.upsample_bilinear(res1, size=self.output_size)
        res2 = F.avg_pool2d(x, self.scale[1], self.stride[1])
        res2_up = F.upsample_bilinear(res2, size=self.output_size)
        res3 = F.avg_pool2d(x, self.scale[2], self.stride[2])
        res3_up = F.upsample_bilinear(res3, size=self.output_size)
        res4 = F.avg_pool2d(x, self.scale[3], self.stride[3])
        res4_up = F.upsample_bilinear(res4, size=self.output_size)
        out = res1_up + res2_up + res3_up + res4_up + x

        return out


class CFFConv2d(nn.Module): # cascade feature fusion
    def __init__(self, f1planes, f2planes, outplane, nclass):
        super(CFFConv2d, self).__init__()
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classifier_conv = nn.Conv2d(f1planes, nclass, kernel_size=1)
        self.conv_dil = nn.Sequential(
            nn.Conv2d(f1planes, outplane, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(outplane),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(f2planes, outplane, kernel_size=1),
            nn.BatchNorm2d(outplane),
        )

    def forward(self, x_f1, x_f2):
        x_f1_up = self.upsample2x(x_f1)
        out1 = self.classifier_conv(x_f1_up)
        x_f1_conv = self.conv_dil(x_f1_up)
        x_f2_conv = self.conv2(x_f2)

        out2 = F.relu(x_f1_conv + x_f2_conv)

        return out1, out2


class ICNet(nn.Module):
    def __init__(self, nclass):
        super(ICNet, self).__init__()

        #self.init_downDim = 32
        #self.init_Dim = 64
        #self.bottle1_downDim = 32
        #self.bottle1_Dim = 128
        #self.bottle2_downDim = 64
        #self.bottle2_Dim = 256
        self.psp_scale = [ (20,15), (12,9), (8, 6), (5, 5) ]
        self.psp_stride = [ (20, 15), (8, 6), (6, 4), (3, 2) ]
        self.psp_oriSize = (20, 15)
        self.nclass = nclass
        # shared weights layers
        self.SWC_Convs = nn.Sequential(
            init_s2(3,32,64),
            nn.MaxPool2d(3,2,padding = 1),
            # conv1_*
            Bottleneck(64, 32, 128),
            Bottleneck(128, 32, 128),
            Bottleneck(128, 32, 128),
            Bottleneck(128, 64, 256, stride=2),
        )

        # f1 dilated convs
        self.F1_Dilated = nn.Sequential(
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 128, 512),

            Bottleneck(512, 128, 512, stride=1, pad=2, dilation=2),
            Bottleneck(512, 128, 512, stride=1, pad=2, dilation=2),
            Bottleneck(512, 128, 512, stride=1, pad=2, dilation=2),
            Bottleneck(512, 128, 512, stride=1, pad=2, dilation=2),
            Bottleneck(512, 128, 512, stride=1, pad=2, dilation=2),
            Bottleneck(512, 256, 1024, stride=1, pad=4, dilation=4),

            Bottleneck(1024, 256, 1024, stride=1, pad=4, dilation=4),
            Bottleneck(1024, 256, 1024, stride=1, pad=4, dilation=4),


            #conv5-3
            psp_pooling(self.psp_scale, self.psp_stride, self.psp_oriSize),

            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
          )

        # f3 convs
        self.F3_Convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # CFF layers
        self.cff_F1 = CFFConv2d(256, 256, 128, self.nclass)
        self.cff_F2 = CFFConv2d(128, 64, 128, self.nclass)

        # upsample
        self.Upsmple_Convs = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, self.nclass, kernel_size=1)
        )

        self.Upsampling4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x, target=None):
        """
        target B*H*W
        """
        x2 = x
        x1 = Variable(x.data[:, :, 0::2, 0::2])

        f2 = self.SWC_Convs(x1)
        x0 = F.avg_pool2d(f2, 3, 2, padding = 1)
        f1 = self.F1_Dilated(x0)
        out1, f_12 = self.cff_F1(f1, f2)

        f3 = self.F3_Convs(x2)
        out2, f_23 = self.cff_F2(f_12, f3)

        out3 = self.Upsmple_Convs(f_23)
        out4 = self.Upsampling4(out3)

        softmax_output = F.softmax(out4)
        # B*C*H*W  -> B*1*H*W
        mask_out = Variable(softmax_output.data.max(1)[1]) # max(1)[0] are max values, max(1)[1] are idxs.  
        if target is not None:
            target1 = Variable(target.data[:, 0::16, 0::16])
            target2 = Variable(target.data[:, 0::8, 0::8])
            target3 = Variable(target.data[:, 0::4, 0::4])

            pairs = { "out1": (out1, target1),
                      "out2": (out2, target2),
                      "out3": (out3, target3),
                      "out4": (out4, target),
                      "mask_out": (mask_out, target)
                    }
            return pairs, self.exports(x, mask_out, target)
        else:
            return self.exports(x, mask_out, target)

    def exports(self, x, mask_out, target):
        result = {"input": x, "output": mask_out}
        if target is not None:
            result['target'] = target.unsqueeze(1)
        return result

if __name__ == '__main__':
    icnet = ICNet()
    icnet.eval()

    x1 = Variable(torch.ones((4, 3, 320, 240)))
#x2 = Variable(torch.rand((4, 3, 640, 480)))
    x2 = Variable(torch.ones((4, 3, 256, 256)))
#x2 = Variable(torch.ones((4, 3, 640, 480)))
#out1, out2, out3, out4 = icnet(x1, x2)
    out1, out2, out3, out4 = icnet(x2)
    print out1
    print out1.size(), out2.size(), out3.size(), out4.size()


