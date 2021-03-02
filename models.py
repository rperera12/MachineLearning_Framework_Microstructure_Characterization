import numpy as np 
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, datasets, models
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
import pathlib
from pathlib import Path
from torch import nn
import cv2
import numpy as np
import time
from IPython.display import clear_output
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import torch
import torch.nn as nn




class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25) 
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.poo2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25) 
        
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.poo3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.5) 
        
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.poo4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.5) 
        
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.dropout5 = nn.Dropout(0.8) 

        self.fc2 = nn.Linear(512, 256)
        self.dropout6 = nn.Dropout(0.25) 
        
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x




class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand        



class ConvNet_hist(nn.Module):
        def __init__(self):
            super(ConvNet_hist, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 12, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.layer2 = nn.Sequential(
                nn.Conv2d(12, 24, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
              
            self.layer3 = nn.Sequential(
                nn.Conv2d(24, 32, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.layer4 = nn.Sequential(
                nn.Conv2d(32, 40, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.fc1 = nn.Linear(40*16*16, 512)
            self.fc2 = nn.Linear(512, 1*53) 


        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = (self.fc2(out))
            return  out



class ConvNet_freq(nn.Module):
        def __init__(self):
            super(ConvNet_freq, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 12, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
 
            self.layer2 = nn.Sequential(
                nn.Conv2d(12, 18, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.fc1 = nn.Linear(18*64*64, 256)
            self.fc2 = nn.Linear(256, 1*53) 

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = (self.fc2(out))
            return  out     



class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.encoder3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.encoder4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv_mid = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),   
                                    nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        
        self.decoder4 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        
        self.decoder3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        
        self.decoder2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        
        self.decoder1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        
        self.upconv4 = nn.ConvTranspose2d(1024, 1024, kernel_size=(2, 2), stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
        self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        self.conv1x1_out = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out = self.encoder1(x)
        out = self.pool1(out)
        out = self.encoder2(out)
        out = self.pool2(out)
        out = self.encoder3(out)
        out = self.pool3(out)
        out = self.encoder4(out)
        out = self.pool4(out)
        out = self.conv_mid(out)
        out = self.upconv4(out)
        out = self.decoder4(out)
        out = self.upconv3(out)
        out = self.decoder3(out)
        out = self.upconv2(out)
        out = self.decoder2(out)
        out = self.upconv1(out)
        out = self.decoder1(out)
        out = self.conv1x1_out(out)

        return out






def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 1024, 3, 1)
        self.conv_last = nn.Conv2d(1024, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        out = self.conv_last(x)

        return out



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()


        ##############-------Begin Encoder--------################
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(3, 64, 9, stride=1, padding=4),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(64, 128, 5, stride=1, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(128, 256, 7, stride=1, padding=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )
        self.conv4 = nn.Sequential(
            torch.nn.Conv2d(256, 512, 9, stride=1, padding=4),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                    )


        ##############-------Begin Decoder--------################  
        self.upconv4 = nn.Sequential(torch.nn.Conv2d(512, 256, 9, stride=1, padding=4),
                            torch.nn.BatchNorm2d(256),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        self.upconv3 = nn.Sequential(torch.nn.Conv2d(256*2, 128, 7, stride=1, padding=3),
                            torch.nn.BatchNorm2d(128),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        self.upconv2 = nn.Sequential(torch.nn.Conv2d(128*2, 64, 5, stride=1, padding=2),
                            torch.nn.BatchNorm2d(64),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        self.upconv1 = nn.Sequential(torch.nn.Conv2d(64*2, 3, 9, stride=1, padding=4),
                            torch.nn.BatchNorm2d(3),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1



