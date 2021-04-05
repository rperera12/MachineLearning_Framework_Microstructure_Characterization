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

import math
import os
from collections import defaultdict
import torch.nn.functional as F


#Defining the GB microstructure image DataLoader
class SteelLoader(Dataset):

    def __init__(self, root_dir, transform = None, pytorch=True):
        super().__init__()
        
        self.pytorch = pytorch
        self.root_dir = root_dir   
        self.transform = transform  

    def __len__(self):
        return int(len(os.listdir(self.root_dir)))

    def __getitem__(self, idx):

        #Note: this line defines the image format to analyze (e.g. JPG,PNG) 
        # Note your images may be of different format (PNG or JPG or TIF)
        image_name = self.root_dir + str(idx) + '_train.png'
        image = np.array(Image.open(image_name)) 

        if self.transform:
            image = self.transform(image) 
        
        sample = {'train_image': image}    

        return sample



#Defining the Classifier Network image DataLoader
class ClassLoader(Dataset):

    def __init__(self, root_dir,  transform = None, pytorch=True):
        super().__init__()
        
        self.pytorch = pytorch
        self.root_dir = root_dir   
        self.transform = transform  

    def __len__(self):
        return 4000 #int(len(os.listdir(self.root_dir))) 

    
    def __getitem__(self, idx):

        # Clever way to create labels, 
        # set 1st class (grains) as even repetitions in directory
        # and 2nd class (particles) as odd repetitions 
        if (idx % 2) == 0:
          if idx>1999:
            idx = idx-2000
          else:  
            idx = idx

          # HINT: Set grain images with trailing _train.png in name
          # HINT: And particle images with no trailing string in name  
          # Note your images may be of different format (PNG or JPG or TIF)
          image_name = self.root_dir + str(idx) + '_train.png'
          image = np.array(Image.open(image_name))
          image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

          labels = 1

        else:
          if idx>1999:
            idx = idx-2000
          else:  
            idx = idx

          # Note your images may be of different format (PNG or JPG or TIF)  
          image_name = self.root_dir + str(idx) + '.png'
          image = np.array(Image.open(image_name))
          image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
          labels = 0

        labels = torch.tensor(labels)    

        if self.transform:
            image = self.transform(image)        

        sample = {'train_image': image, 'label': labels}    

        return sample



#Defining the RGB Segmentation Network image DataLoader
class RGBLoader(Dataset):

    def __init__(self, root_dir, seg_dir, transform = None, pytorch=True):
        super().__init__()
        
        self.pytorch = pytorch
        self.root_dir = root_dir
        self.seg_dir = seg_dir     
        self.transform = transform  

    def __len__(self):
        return int(len(os.listdir(self.root_dir)))    

    
    def __getitem__(self, idx):

        # Note your images may be of different format (PNG or JPG or TIF)
        image_name = self.root_dir + str(idx) + '_train.png'
        image = np.array(Image.open(image_name))
                
        # Note your images may be of different format (PNG or JPG or TIF)
        mask_name = self.seg_dir + str(idx) + '_label.png'
        mask_image = np.array(Image.open(mask_name))
        mask_image = mask_image.transpose((2, 0, 1))/255 

        mask_image = torch.tensor(mask_image)    

        if self.transform:
            image = self.transform(image) 

        sample = {'train_image': image, 'mask_image': mask_image}    

        return sample        



#Defining the Histogram Prediction Network DataLoader
class HistLoader(Dataset):

    def __init__(self, seg_dir, hist_dir, transform = None, pytorch=True):
        super().__init__()
        
        self.pytorch = pytorch
        self.hist_dir = hist_dir
        self.seg_dir = seg_dir     
        self.transform = transform  

    def __len__(self):
        return int(len(os.listdir(self.seg_dir)))   

    
    def __getitem__(self, idx):

        # Note your images may be of different format (PNG or JPG or TIF)
        image_name = self.seg_dir + str(idx) + '_label.png'
        image = np.array(Image.open(image_name))
        image = cv2.resize(image, (256,256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY ) 
                
        hist_name = self.hist_dir + str(idx) + '_hist.txt'

        with open(hist_name) as file:
          array2d = [[np.float32(digit) for digit in line.split()] for line in file]
        
        hist_array = np.array(array2d)
        total_hist_array = np.zeros([2,53])
        _ , column = hist_array.shape
        total_hist_array[0,0:column] = hist_array[0,:]
        total_hist_array[1,0:column] = hist_array[1,:]

        if column < 53:

          dif = 53-column
          i = 1
          j = 0
          while i < dif:
            total_hist_array[0,column+i] = 0 
            total_hist_array[1,column+i] = 0
            i = i + 1
            j = j + 1

        total_hist_array[0,:] /= 50 # Normalize radii
        total_hist_array[0,:] = np.round(total_hist_array[0,:],1)

        total_hist_array[1,:] *= 100 # Normalize frequency
        total_hist_array[1,:] = np.round(total_hist_array[1,:],1)

        total_hist_array = torch.tensor(total_hist_array)    

        if self.transform:
            image = self.transform(image) 
        
        sample = {'train_image': image, 'hist_array': (total_hist_array)}    
          
        return sample              




class BinaryLoader(Dataset):
    def __init__(self, root_dir, gt_dir, pytorch=True):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.root_dir = root_dir
        self.gt_dir = gt_dir
        self.pytorch = pytorch
                                    
    def __len__(self):
        
        return int(len(os.listdir(self.root_dir)))
     
    def __getitem__(self, idx):
        
        # Note your images may be of different format (PNG or JPG or TIF)
        image_name = self.root_dir + str(idx) + '_train.jpg'
        image = np.array(Image.open(image_name))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        x = image.transpose((2,0,1))/255
        x = torch.tensor(x)

        # Note your images may be of different format (PNG or JPG or TIF)
        mask_name = self.gt_dir + str(idx) + '_Binary.png'
        mask = np.array(Image.open(mask_name))
        for k in range(256):
            for j in range(256):
                if mask[k,j] == True:
                    mask[k,j] = 0
                elif mask[k,j] == False:
                    mask[k,j] = 1    
        #mask = mask.transpose((2,0,1))/255
        #mask = mask[0,:,:]
        y = torch.tensor(mask)
        
        return x, y