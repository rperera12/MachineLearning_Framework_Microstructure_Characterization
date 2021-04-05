import numpy as np 
import pandas as pd
import time
import pathlib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, datasets, models
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch import nn
import cv2
import torch.optim as optim
import copy

# Allow IPython for Google Colab
#from IPython.display import clear_output

import os
from collections import defaultdict
import torch.nn.functional as F

from utils import *
from dataloaders import RGBLoader
from config import args_generator


args = args_generator()

# Load Paths for training grain images and their labels
train_path = os.getcwd() + args.train_dir
seg_path = os.getcwd() + args.gt_dir

# Define transformation for RGB network
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

# Load all images with their segmentation labels
data = RGBLoader(train_path, 
                    seg_path, transform = trans)

if args.visualize:
    # Test plotting a train image with its label
    pic_id = np.random.randint(3560)
    test_sample = data[pic_id]
    x = (test_sample['train_image']).numpy()
    y = (test_sample['mask_image']).numpy()

    x_p = untransform((test_sample['train_image']))
    y_p = y.transpose((1, 2, 0)) 

    fig, ax = plt.subplots(1,2, figsize=(10,10))
    ax[0].imshow(x_p)
    ax[1].imshow(y_p)


# Set up batches for Train-Set, Valid-Set, and Test-Set
train_ds, valid_test_ds = torch.utils.data.random_split(data, (args.train_set, args.valid_set*2))
train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)

valid_ds, test_ds = torch.utils.data.random_split(valid_test_ds, (args.valid_set, args.test_set))
valid_dl = DataLoader(valid_ds, batch_size=10, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=10, shuffle=False)


# Define the RGB Seg Network network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.ML_network == 'DENSE-UNet': 
    from models import UNet
    model = UNet()
elif args.ML_network == 'ResNet-UNet':
    from models import ResNetUNet
    model = ResNetUNet(3)
elif args.ML_network == 'U-Net':
    from models import unet
    model = unet()     

model = model.to(device)


# Set A function for training 
def train_model(model, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    train_loss, valid_loss = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                dataloaders = train_dl
                model.train()  

            else:
                dataloaders = valid_dl
                model.eval()  

            measure = defaultdict(float)
            epoch_no = 0

            for sample in dataloaders:
                if torch.cuda.is_available():
                    inputs = (sample['train_image']).type(torch.cuda.FloatTensor)
                    labels = (sample['mask_image']).type(torch.cuda.FloatTensor)
                else:
                    inputs = (sample['train_image']).type(torch.FloatTensor)
                    labels = (sample['mask_image']).type(torch.FloatTensor)
                #inputs = inputs.to(device)
                #labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, measure)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_no += inputs.size(0)

            print_measure(measure, epoch_no, phase)
            epoch_loss = measure['loss'] / epoch_no

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)    

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    torch.save(model, (os.getcwd() + args.out_dir))
    return model, train_loss, valid_loss


optimizer_f = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
model, train_loss, valid_loss = train_model(model, optimizer_f, num_epochs=args.n_epochs)


if args.visualize:

    plt.figure(1, figsize=(10,8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.show()

    model.eval()  

    sample_b = next(iter(test_dl))
    xb = sample_b['train_image']
    yb = sample_b['mask_image']
    inputs = xb.to(device)
    labels = yb.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()

    x_pp = xb[4,:,:,:]
    x_pp = untransform(x_pp)
    plt.figure(2)
    plt.imshow(x_pp)
    plt.show()

    yy = pred[4,:,:,:].transpose((1, 2, 0))
    plt.figure(3)
    plt.imshow(yy)
    plt.show()


