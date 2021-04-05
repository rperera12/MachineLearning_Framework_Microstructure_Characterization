import numpy as np 
import pandas as pd
import pathlib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, datasets, models
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
from torch import nn
import cv2

# Allow IPython for Google Colab
#from IPython.display import clear_output

import math
import os
from collections import defaultdict
import torch.nn.functional as F


# Dice Loss Function, used for Encoder-Decoder Networks
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(out, label, measure, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(out, label)
    out = torch.sigmoid(out)
    dice = dice_loss(out, label)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    measure['bce'] += bce.data.cpu().numpy() * label.size(0)
    measure['dice'] += dice.data.cpu().numpy() * label.size(0)
    measure['loss'] += loss.data.cpu().numpy() * label.size(0)

    return loss


def print_measure(measure, epoch_no, phase):
    outs = []
    for k in measure.keys():
        outs.append("{}: {:4f}".format(k, measure[k] / epoch_no))

    print("{}: {}".format(phase, ", ".join(outs)))


# Untransform function for Encoder-Decoder
def untransform(img):
    img = img.numpy().transpose((1, 2, 0))    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def acc_metric(predb, yb):       
    return (predb.argmax(dim=1) == yb).float().mean()


def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()


def imshow(img):
    # unnormalize
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


