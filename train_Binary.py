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
from IPython.display import clear_output
import os

from utils import BinaryLoader, acc_metric, predb_to_mask
from models import UNET


count = 0
main_dir = os.getcwd() + '/data/pore_particle/train/'
gt_dir = os.getcwd() + '/data/pore_particle/seg_label/'

data = BinaryLoader(main_dir, gt_dir)
print(len(data))

xx, yy = data[150]
xx = xx.numpy()
yy = yy.numpy()

xx = xx.transpose((1,2,0))

fig, ax = plt.subplots(1,2, figsize=(10,9))
ax[0].imshow(xx)
ax[1].imshow(yy)
plt.show()

train_ds, valid_test_ds = torch.utils.data.random_split(data, (1800, 248))
train_dl = DataLoader(data, batch_size=12, shuffle=False)

valid_ds, test_ds = torch.utils.data.random_split(valid_test_ds, (124, 124))
valid_dl = DataLoader(valid_ds, batch_size=12, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=12, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNET(3,2)
unet.to(device)


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

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
                x = x.type(torch.cuda.FloatTensor) 
                y = y.type(torch.cuda.LongTensor) 
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

            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    torch.save(unet.state_dict(),'./Hopper_Test.pt')
    return train_loss, valid_loss    


    
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(unet.parameters(), lr=0.01)
train_loss, valid_loss = train(unet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=40)

unet.load_state_dict(torch.load('./Pretrained_weights/simple_unet_weights.pt', map_location=torch.device('cpu')))

img_o = np.array(Image.open('data/particle_pore/train/2000.png'))
img_o = cv2.cvtColor(img_o, cv2.COLOR_RGBA2RGB)
img = ToTensor()(img_o).unsqueeze(0)
img = Variable(img)
x1 = img.type(torch.cuda.FloatTensor)

pred22 = unet(x1)

plt.subplot(121)
plt.imshow(img_o)
plt.subplot(122)
plt.imshow(predb_to_mask(pred22, 0))
plt.show()