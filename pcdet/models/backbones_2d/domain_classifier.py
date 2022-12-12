#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class pool_1(nn.Module):
    ''' Follow DANN github
    '''

    def __init__(self):
        super().__init__()

        layers = nn.Sequential()
        layers.add_module('d_fc1', nn.Linear(512, 128))
        layers.add_module('d_bn1', nn.BatchNorm1d(128))
        layers.add_module('d_relu1', nn.ReLU(True))
        layers.add_module('d_fc2', nn.Linear(128, 2))
        layers.add_module('d_softmax', nn.LogSoftmax(dim=1))
        self.layers = layers

    def forward(self, feat, alpha):
        B, C, W, H = feat.shape
        feat = feat.reshape(B, C, -1)
        feat = torch.max(feat, dim=2)[0]
        feat = ReverseLayerF.apply(feat, alpha)
        domain_output = self.layers(feat)

        return domain_output

class conv_1(nn.Module):
    ''' Follow Adaptive Teacher
    '''
    def __init__(self):
        super().__init__()

        layers = nn.Sequential()
        layers.add_module('conv1', nn.Conv2d(512, 256, kernel_size=3, padding=1))
        layers.add_module('leaky_relu1', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.add_module('conv2', nn.Conv2d(256, 128, kernel_size=3, padding=1))
        layers.add_module('leaky_relu2', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.add_module('conv3', nn.Conv2d(128, 128, kernel_size=3, padding=1))
        layers.add_module('leaky_relu3', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.add_module('conv4', nn.Conv2d(128, 1, kernel_size=3, padding=1))
        self.layers = layers

    def forward(self, feat, alpha=None):
        feat = grad_reverse(feat)
        feat = self.layers(feat)
        return feat

model_dict = {'POOL_1': pool_1,
              'CONV_1': conv_1
              }
