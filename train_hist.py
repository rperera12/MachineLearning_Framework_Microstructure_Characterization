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
#from IPython.display import clear_output
#from google.colab.patches import cv2_imshow
import os
from collections import defaultdict
import torch.nn.functional as F

from models import ConvNet_freq, ConvNet_hist
from dataloaders import HistLoader
from config import args_generator

args = args_generator()


train_path = args.train_dir + 'seg_label/'
hist_path =  args.train_dir + 'histogram_label/'

trans = transforms.Compose([
    transforms.ToTensor()]) 

data = HistLoader(train_path, 
                    hist_path, transform = trans)


if args.visualize:
  print(len(data))

  test_sample = data[8]
  x = (test_sample['train_image']).numpy()
  y_hist = (test_sample['hist_array']).numpy()

  x_p = x.transpose((1,2,0))
  y_h = y_hist 

  plt.figure(figsize = (10,10))
  plt.scatter(np.arange(len(y_h[1,:])), y_h[1,:])
  plt.xlabel('Range of Frequencies')
  plt.ylabel('Relative Frequency')
  plt.show()

  plt.figure(figsize = (10,10))
  plt.scatter(np.arange(len(y_h[0,:])), y_h[0,:])
  plt.xlabel('Range of Radius')
  plt.ylabel('Equivalent Grain Radius (micron)')
  plt.show()

  plt.figure(figsize = (10,10))
  plt.bar(y_h[0,:], y_h[1,:], width = 0.3)
  plt.xlabel('Equivalent Grain Radius (micron)')
  plt.ylabel('Relative Frequency')
  plt.title('Histogram Diplay')
  plt.show()



train_ds, valid_test_ds = torch.utils.data.random_split(data, (args.train_set, args.valid_set*2))
train_dl = DataLoader(train_ds, batch_size=22, shuffle=False)

valid_ds, test_ds = torch.utils.data.random_split(valid_test_ds, (args.valid_set, args.test_set))
valid_dl = DataLoader(valid_ds, batch_size=12, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=12, shuffle=True)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


model_rad = ConvNet_hist()
model_rad = model_rad.to(device)

model_freq = ConvNet_freq()
model_freq = model_freq.to(device)



learning_rate = 0.001
criterion = nn.SmoothL1Loss() # Huber Loss Funtion
optimizer_rad = torch.optim.Adam(model_rad.parameters(), lr=learning_rate) 
optimizer_freq = torch.optim.Adam(model_freq.parameters(), lr=learning_rate) 
        
def train_net(n_epochs,phases,optimizer_name):

    # prepare the net for training
    if optimizer_name == 'optimizer_rad':
      optimizer = optimizer_rad 
      model_name = './Pretrained_weights/histogram_rad_weights.pth'
      model = model_rad
    elif optimizer_name == 'optimizer_freq':
      optimizer = optimizer_freq
      model_name = './Pretrained_weights/histogram_freq_weights.pth'
      model = model_freq
    
    for phase in phases:
      if phase == 'train':
        model.train()
        dataloader = train_dl
      else:
        model.eval()  
        dataloader = valid_dl
          
      for epoch in range(n_epochs):  # loop over the dataset multiple times
      
          running_loss = 0.0

          # train on batches of data, assumes you already have train_loader
          for batch_i, data in enumerate(dataloader):
              # get the input images and their corresponding labels
              
              initial = data['train_image'] 
                
              if optimizer_name == 'optimizer_rad':  
                labels_hist = data['hist_array']
                labels = (labels_hist[:,0,:])

              elif optimizer_name == 'optimizer_freq':  
                labels_hist = data['hist_array']
                labels = (labels_hist[:,1,:])  

              # flatten pts
              labels = labels.view(labels.size(0), -1)

              if torch.cuda.is_available():
                labels = labels.type(torch.cuda.FloatTensor)
                initial = initial.type(torch.cuda.FloatTensor)
              else:  
                labels = labels.type(torch.FloatTensor)
                initial = initial.type(torch.FloatTensor)
                
              # forward pass to get outputs
              output_pts = model(initial)

              #outputs = torch.round(output_pts)
              # calculate the loss between predicted and target keypoints
              loss = criterion(output_pts, labels)

              print('already good to go')

              if phase == 'train': 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

              running_loss += loss.item()
              if batch_i % 22 == 21:    # print every 10 batches
                  print('Phase: {}, Epoch: {}, Batch: {}, Avg. Loss {}'.format(phase, epoch + 1, batch_i+1, running_loss/22))
                  running_loss = 0        
      if phase == 'train':
        torch.save(model,model_name)



#Training
phases = ['train','valid']
for i in range(2):
  if i == 0:
    #print('Skip relative radius training')
    train_net(args.n_epochs,phases, 'optimizer_rad')

  elif i == 1:  
    #print('Skip relative frequency training')
    train_net(args.n_epochs,phases, 'optimizer_freq')  



if args.visualize:      

  model_rad = torch.load(args.out_dir + '_rad_weights.pth')
  model_freq = torch.load(args.out_dir +'_freq_weights.pth')


  results_sample = next(iter(train_dl))
  inn = results_sample["train_image"].type(torch.cuda.FloatTensor)
  outt_hist = results_sample["hist_array"].cpu().numpy()

  orig_rad = outt_hist[4,0,:]
  orig_freq = outt_hist[4,1,:]


  with torch.no_grad():
      pred_rad = model_rad(inn)
      pred_rad = F.relu(pred_rad)    

  with torch.no_grad():
      pred_freq = model_freq(inn)
      pred_freq = F.relu(pred_freq)   

  pred_rad = pred_rad[4,:].cpu().numpy()
  pred_freq = pred_freq[4,:].cpu().numpy()


  np.round(pred_rad,1)
  np.round(orig_rad,1)

  np.round(orig_freq,2)

  plt.figure(figsize = (10,10))
  plt.bar((orig_rad), orig_freq, width = 0.2)
  plt.xlabel('Equivalent Grain Radius (micron)')
  plt.ylabel('Relative Frequency')
  plt.title('PSILM Labeled Results')
  plt.show()

  plt.figure(figsize = (10,10))
  plt.bar(pred_rad, pred_freq, width = 0.2)
  plt.xlabel('Equivalent Grain Radius (micron)')
  plt.ylabel('Relative Frequency')
  plt.title('ConvNet() Predicted Results')
  plt.show()

  barWidth = 0.15
  plt.figure(figsize = (40,40))
  plt.bar((orig_rad), orig_freq, width = 0.1, alpha = 0.5, linewidth = 0, edgecolor='white', label = 'PSILM Results')
  plt.bar(pred_rad, pred_freq, width = 0.1, alpha = 0.5, linewidth = 0, edgecolor='white', label = 'ConvNet() Results')
  plt.xlabel('Equivalent Grain Radius (micron)')
  plt.ylabel('Relative Frequency')
  plt.legend()
  plt.show()


