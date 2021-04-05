import sys

import numpy as np 
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
import pathlib
from pathlib import Path
from torch import nn
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import cv2
import numpy as np
import time

# Allow IPython for Google Colab
#from IPython.display import clear_output
import os
import argparse

from config import args_generator
from dataloaders import BinaryLoader
from utils import acc_metric, predb_to_mask
from models import UNET


args = args_generator()

main_dir = os.getcwd() + args.train_dir
gt_dir = os.getcwd() + args.gt_dir

data = BinaryLoader(main_dir, gt_dir)

if args.visualize:
    pic_id = np.random.randint(500)
    xx, yy = data[pic_id]
    xx = xx.numpy()
    yy = yy.numpy()
    xx = xx.transpose((1,2,0))

    fig, ax = plt.subplots(1,2, figsize=(10,9))
    ax[0].imshow(xx)
    ax[1].imshow(yy)
    plt.show()

train_ds, valid_test_ds = torch.utils.data.random_split(data, (args.train_set, args.valid_set*2))
train_dl = DataLoader(train_ds, batch_size=12, shuffle=True)

valid_ds, test_ds = torch.utils.data.random_split(valid_test_ds, (args.valid_set, args.test_set))
valid_dl = DataLoader(valid_ds, batch_size=12, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=12, shuffle=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.ML_network == 'Simple-UNet':
    model = UNET(3,2)
    model.to(device)


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()

    train_loss, valid_loss = [], []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  
                dataloader = train_dl
            else:
                model.train(False)  
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            for x, y in dataloader:
                if torch.cuda.is_available():
                    x = x.type(torch.cuda.FloatTensor) 
                    y = y.type(torch.cuda.LongTensor) 
                else:   
                    x = x.type(torch.FloatTensor) 
                    y = y.type(torch.LongTensor) 
                step += 1

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

                if step % 100 == 0:
                    print('Current step: {}  Loss: {}  Acc: {} '.format(step, loss, acc))

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            #clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    torch.save(model.state_dict(), args.out_dir)
    return train_loss, valid_loss    


    
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
train_loss, valid_loss = train(model, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=args.n_epochs)


if args.visualize:

    model.load_state_dict(torch.load(args.out_dir, map_location=torch.device('cpu')))

    # Note: you can select random microstructure from 
    # internet source and test it here
    pic_valid = np.random.randint(124)
    img_o = np.array(Image.open(main_dir + str(pic_valid) + '.png'))
    img_o = cv2.cvtColor(img_o, cv2.COLOR_RGBA2RGB)
    img = ToTensor()(img_o).unsqueeze(0)
    img = Variable(img)
    
    if torch.cuda.is_available():
        x1 = img.type(torch.cuda.FloatTensor)
    else:
        x1 = img.type(torch.FloatTensor)    
    pred22 = model(x1)
    plt.subplot(121)
    plt.imshow(img_o)
    plt.subplot(122)
    plt.imshow(predb_to_mask(pred22, 0))
    plt.show()