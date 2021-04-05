import os
import sys
import copy

import argparse
import numpy as np
import torch

import argparse

parser = argparse.ArgumentParser()

### build arguments
parser.add_argument('--Application', default='Binary_Segmentation')
parser.add_argument('--ML_network',  default='Simple-UNet')
parser.add_argument('--train_dir', default='/data/pore_particle/train/', help="Training dataset directory")
parser.add_argument('--gt_dir', default='/data/pore_particle/seg_label/', help="Labeled dataset directory")
parser.add_argument('--train_set', type=float, default=10, help = 'Training set images')
parser.add_argument('--valid_set', type=float, default=10, help = 'Validation set images')
parser.add_argument('--test_set', type=float, default=10, help = 'Test set images')
parser.add_argument('--n_epochs', type=int, default=20, help = 'Number of Epochs')
parser.add_argument('--visualize', type=int, default=0, help = 'Visualize Output')
parser.add_argument('--out_dir', default='/Pretrained_weights/simple_unet_weights.pt')


#/home/rob/Desktop/Laptop_rob/New_Github_Repo/data/pore_particle/train

def args_generator():
    args = parser.parse_args()
    #args , unknown = parser.parse_known_args()
    if args.Application == 'Binary_Segmentation':
        args.train_set = 1800
        args.valid_set = 124
        args.test_set = 124
            

    elif args.Application == 'RGB_Segmentation':
        txt =  input('Select RGB Segmentation ML Network to train (options: U-Net, ResNet-UNet, DENSE-UNet)')
        
        args.ML_network = txt
        args.train_dir = '/data/grain/train/'
        args.gt_dir = '/data/grain/seg_label/'

        args.train_set = 3560
        args.valid_set = 108
        args.test_set = 108

        args.out_dir = '/Pretrained_weights/DENSE_UNet_weights.pth'


    elif args.Application == 'Classification':
        args.train_dir = '/data/classifier_pores_grains/'
        args.out_dir = '/Pretrained_weights/pore_vs_grain_weights.pth'
        args.ML_network = 'Net'
        args.train_set = 3800
        args.valid_set = 100
        args.test_set = 100


    elif args.Application == 'Grain_Distribution':
        args.train_dir = 'data/grain/' 
        args.out_dir = './Pretrained_weights/histogram'
        args.train_set = 3560
        args.valid_set = 108
        args.test_set = 108

    return args