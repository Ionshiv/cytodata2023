from __future__ import absolute_import, division, print_function, unicode_literals

# Pytorch and Pytorch Geometric
import torch as tch
import torch.nn as nn
import torch.optim as optim


from torch_geometric.nn import GCNConv, GATConv, summary as gsummary, global_mean_pool, global_max_pool
from torch_geometric.data import Data#, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dropout_adj

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

#Internal Libraries
from graphmake import GraphMake

# External Helper libraries
from torchsummary import summary as asummary
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# self.device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
# device = tch.device('cuda')

class simpleGNN(nn.Module):
    def __init__(self, input_dim, device=None):
        if device:
            self.device = device
        else:
            self.device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
        # simpleGNN Layers
        super(simpleGNN, self).__init__()
        self.conv1 = GATConv(input_dim, 32, heads=1, concat=True)  # Single attention head
        self.conv2 = GATConv(32, 32, heads=2, concat=True)  # Single attention head
        self.conv3 = GATConv(64, 64, heads=2, concat=True)
        self.conv4 = GATConv(128, 32, heads=2, concat=True)  # Single attention head
        self.conv5 = GATConv(64, 8, heads=2, concat=True)  # Single attention head  # Single attention head
        self.fc3 = nn.Linear(16, 1)  # Output layer with 1 node
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv4(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv5(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.fc3(x)
        return x

    def fitGNN(self, t_loader, v_loader, num_epochs, optimizer, criterion, scheduler):
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            # Training Phase
            self.train()
            train_loss_items = []
            for batch in t_loader:
                batch.to(self.device)
                optimizer.zero_grad()
                # Use Batch Data object in forward pass
                outputs = self(batch.x.float(), batch.edge_index, batch.batch)
                loss = criterion(outputs, batch.y)

                l1_lambda = 0.0001
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss += l1_lambda * l1_norm

                loss.backward()
                optimizer.step()
                train_loss_items.append(loss.item())

            avg_train_loss = sum(train_loss_items) / len(train_loss_items)
            train_losses.append(avg_train_loss)
            # Validation Phase (assuming you have a separate validation loader)
            self.eval()
            val_loss_items = []
            with tch.no_grad():
                for val_batch in v_loader:
                    val_batch.to(self.device)
                    val_outputs = self(val_batch.x.float(), val_batch.edge_index, val_batch.batch)
                    val_loss = criterion(val_outputs, val_batch.y)
                    val_loss_items.append(val_loss.item())

            avg_val_loss = sum(val_loss_items) / len(val_loss_items)
            val_losses.append(avg_val_loss)
            if epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {avg_val_loss:.4f}')
            elif epoch == num_epochs:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {avg_val_loss:.4f}')
            scheduler.step()
        return self, train_losses, val_losses
    
    def plot_history(train_losses, val_losses, model_name):
        fig = plt.figure(figsize=(15, 5), facecolor='w')
        ax = fig.add_subplot(121)
        ax.plot(train_losses)
        ax.plot(val_losses)
        ax.set(title=model_name + ': Model loss', ylabel='Loss', xlabel='Epoch')
        ax.legend(['Train', 'Test'], loc='upper right')
        ax = fig.add_subplot(122)
        ax.plot(np.log(train_losses))
        ax.plot(np.log(val_losses))
        ax.set(title=model_name + ': Log model loss', ylabel='Log loss', xlabel='Epoch')
        ax.legend(['Train', 'Test'], loc='upper right')
        plt.savefig('simpleGNNoutput.png')
        plt.show()
        plt.close()

    def weights_init(m):
        if isinstance(m, (GCNConv, GATConv)):
            nn.init.xavier_uniform_(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)