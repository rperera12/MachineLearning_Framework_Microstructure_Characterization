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
import torch.optim as optim

# Allow IPython for Google Colab
#from IPython.display import clear_output
import os

from utils import imshow
from dataloaders import ClassLoader
from config import args_generator


args = args_generator()

# Define the class images directory
train_path = os.getcwd() + args.train_dir

# Define transformation for Classifier Network
trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.CenterCrop(256),
     transforms.Normalize((0.5), (0.5))])

# Load all images using the ClassLoader Class
data = ClassLoader(train_path, 
                    transform = trans)


if args.visualize:
    #Test a single output
    pic_id = np.random.randint(args.train_set)
    test_sample = data[pic_id]
    x = (test_sample['train_image']).numpy()
    y = (test_sample['label']).numpy()
    print(x.shape) 
    print(y.shape)

    classes = ('pore' , 'grain')

    x_p = x.transpose((1, 2, 0))
    plt.imshow(x_p[:,:,0])
    plt.title(classes[int(y)]) 



# Set up Batches for training 
# Note: You can increase this batch = 10 if needed
train_ds, valid_test_ds = torch.utils.data.random_split(data, (args.train_set, args.valid_set*2))
train_dl = DataLoader(train_ds, batch_size=10, shuffle=False)

valid_ds, test_ds = torch.utils.data.random_split(valid_test_ds, (args.valid_set, args.test_set))
valid_dl = DataLoader(valid_ds, batch_size=10, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=10, shuffle=False)

sample_b = next(iter(train_dl))


if args.visualize:
    # get some random training images
    
    images = dataiter['train_image']
    labels =  dataiter['label']
    print(labels)

    # plot images
    imshow(torchvision.utils.make_grid(images))


# Assuming you will be using GPU, else change accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.ML_network == 'Net':
    from models import Net
net = Net().to(device)


# Define Optimization and Loss Functions 
# Note: You can try tesing different learning rates 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Begin Training, you can Change EPOCH if needed
EPOCH = args.n_epochs
for epoch in range(EPOCH):  # 

    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
    #i = 0
    #for data in train_dl:
        # I am assuing you will be using GPU, in any case remove cuda below
        if torch.cuda.is_available():
            inputs = data['train_image'].type(torch.cuda.FloatTensor)
            labels =  data['label'].type(torch.cuda.LongTensor)
        else:
            inputs = data['train_image'].type(torch.FloatTensor)
            labels =  data['label'].type(torch.LongTensor)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass, backward pass, and optimize
        outputs = net(inputs)
        print('Already did one pass', i)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # print every 10 batches
        if i % 10 == 9:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
        #i+=1

print('Finished Training')

# Save the trained wights
PATH = '.' + os.getcwd() + args.out_dir
torch.save(net.state_dict(), PATH)


if args.visualize:
    # Now let's test the performance of the trained network
    net.load_state_dict(torch.load(PATH))
    
    # Load the test set
    dataiter = next(iter(test_dl))
    images = dataiter['train_image']
    labels = dataiter['label']

    # Only plotting the first 4 images from the batch
    images = images[0:4,:,:,:]

    # Make predictions about their class
    outputs = net(images)

    # plot images
    imshow(torchvision.utils.make_grid(images))

    # Print the Ground Truth Labels
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Print their predicted Labels to compare to Ground Truth
    print('Predicted: ', ' '.join('%5s' % classes[outputs[j]]
                                for j in range(4)))


