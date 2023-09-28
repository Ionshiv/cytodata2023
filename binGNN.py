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
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# self.device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
# device = tch.device('cuda')

class binGNN(nn.Module):
    def __init__(self, input_dim, model_name='default', device=None):
        self.model_name = model_name
        if device:
            self.device = device
        else:
            self.device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
        # simpleGNN Layers
        super(binGNN, self).__init__()
        self.conv1 = GATConv(input_dim, 32, heads=1, concat=True)  # Single attention head
        self.conv2 = GATConv(32, 32, heads=2, concat=True)  # Single attention head
        self.conv3 = GATConv(64, 64, heads=2, concat=True)
        self.conv4 = GATConv(128, 32, heads=2, concat=True)  # Single attention head
        self.conv5 = GATConv(64, 8, heads=2, concat=True)  # Single attention head  # Single attention head
        self.fc3 = nn.Linear(16, 1)  # Output layer with 1 node
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
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
        x = self.sigmoid(x)
        return x

    def fitGNN(self, t_loader, v_loader, num_epochs, optimizer, criterion, scheduler):
        self.train()
        self.weights_init()
        self.t_loader = t_loader
        self.v_loader = v_loader
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
            if epoch % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {avg_val_loss:.4f}')
            elif epoch == int(num_epochs-1):
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {avg_val_loss:.4f}')
            scheduler.step()
        return self, train_losses, val_losses

    def print_eval(self, fold=0):
        self.eval()
        y_true = []
        y_score = []
        # Forward pass through the network and gather predictions and true labels
        with tch.no_grad():
            for batch in self.v_loader:
                data = batch.to(self.device)
                preds = self(data.x, data.edge_index, data.batch)
                preds = tch.sigmoid(preds)  # Apply sigmoid if not included in the model
                y_score.extend(preds.cpu().numpy())
                y_true.extend(data.y.cpu().numpy())

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        lw = 2  # Line width
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (AUC={roc_auc})')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(f'output/{self.model_name}_{fold}_roc_auc.png')
        plt.close

        # Compute and print confusion matrix
        y_pred = [1 if s >= 0.5 else 0 for s in y_score]
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:')
        print(cm)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            title='Confusion Matrix',
            ylabel='True label',
            xlabel='Predicted label')
        plt.savefig(f'output/{self.model_name}_{fold}_confusion_matrix.png')
        plt.close()

    def plot_history(self, train_losses, val_losses, strarg=''):
        fig = plt.figure(figsize=(15, 5), facecolor='w')
        ax = fig.add_subplot(121)
        ax.plot(train_losses)
        ax.plot(val_losses)
        ax.set(title=f'{self.model_name}  {strarg}  : Model loss', ylabel='Loss', xlabel='Epoch')
        ax.legend(['Train', 'Test'], loc='upper right')
        ax = fig.add_subplot(122)
        ax.plot(np.log(train_losses))
        ax.plot(np.log(val_losses))
        ax.set(title=self.model_name + ': Log model loss', ylabel='Log loss', xlabel='Epoch')
        ax.legend(['Train', 'Test'], loc='upper right')
        plt.savefig(f'output/{self.model_name}_{strarg}.png')
        plt.close()

    def weights_init(self):
        if isinstance(self, (GCNConv, GATConv)):
            nn.init.xavier_uniform_(self.weight.data)
        elif isinstance(self, nn.Linear):
            nn.init.xavier_uniform_(self.weight.data)
        else:
            raise ValueError()