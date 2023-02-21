import numpy as np
import torch as tch
from torch import nn as nn
from torch.nn import functional as f
from torch import optim as optim
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets

class simpleCAE(nn.Module):
    def __init__(self):
        print('constructing simpleCAE')
        super(simpleCAE, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)


    def forward(self, x):
         # add hidden layers with relu activation function
        # and maxpooling after
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = f.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = f.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = tch.sigmoid(self.t_conv2(x))
                
        return x
        



