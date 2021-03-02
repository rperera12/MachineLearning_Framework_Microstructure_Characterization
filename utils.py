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
import torch
import torch.nn as nn
from torchvision import models
import math



#Defining the microstructure image DataLoader
class SteelLoader(Dataset):

    def __init__(self, root_dir, transform = None, pytorch=True):
        super().__init__()
        
        self.pytorch = pytorch
        self.root_dir = root_dir   
        self.transform = transform  

    def __len__(self):
        return 3776    

    def __getitem__(self, idx):

        #Note: this line defines the image format to analyze (e.g. JPG,PNG) 
        image_name = self.root_dir + str(idx) + '_train.png'
        image = np.array(Image.open(image_name)) 

        if self.transform:
            image = self.transform(image) 
        
        sample = {'train_image': image}    

        return sample




