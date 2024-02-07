import os
import gc
import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import datasets, transforms, models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        mod = models.vgg16(weights = 'VGG16_Weights.IMAGENET1K_V1')
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = mod.features[:23]
        self.backend = self.make_layers(self.backend_feat, in_channels = 512, dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1, padding ='same')
        self.backend.apply(self.init_weights)
        self.output_layer.apply(self.init_weights)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

         
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # nn.init.normal_(m.weight, 0.01)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        else:
            pass

            
    def make_layers(self, cfg, in_channels = 3,batch_norm=False,dilation = False):

        if dilation:
            d_rate = 2
        else:
            d_rate = 1

        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding ='same',dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
