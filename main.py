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
from SimpleCAE import simpleCAE as scae

def main():
    print('starting run')
    device = tch.device('cuda')
    tensorTF = transforms.ToTensor()
    # dataset = datasets.MNIST(root = "./mnistDATA", train = True, download=True, transform=tensorTF)
    train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=tensorTF)
    test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=tensorTF)                          
    train_loader = DataLoader(dataset = train_data, batch_size = 32, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=32, num_workers=0)


    model = scae()
    print(model)
    training(model, train_loader)


def training(model, train_loader):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    epochs = 5
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        for data in train_loader:
            images, _ = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch, train_loss))



if __name__ == "__main__":
    main()