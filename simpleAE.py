from __future__ import absolute_import, division, print_function, unicode_literals

# Pytorch
import torch as tch
import torch.nn as nn
import torch.optim as optim

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

class simpleAE(nn.Module):
    def __init__(self, input_dim, model_name='default', device=None):
        super(simpleAE, self).__init__()
        self.model_name = model_name
        print('model init')
        if device:
            self.device = device
        else:
            self.device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
        self.apply(self.weights_init)
        # Encoder
        self.enc_conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv2 = nn.Conv2d(64, input_dim, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        print('forward')
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))

        # Decoder with skip-connection
        x = self.relu(self.dec_conv1(x) + x)
        x = self.sigmoid(self.dec_conv2(x))

        return x

    def fitAE(self, t_loader, v_loader, num_epochs, optimizer, criterion, scheduler):
        self.train()
        train_losses = []
        val_losses = []
        print('fitting')
        for epoch in range(num_epochs):
            # Training Phase
            self.train()
            train_loss_items = []
            for batch in t_loader:
                img = batch['image'].to(self.device)
                optimizer.zero_grad()
                # Forward pass
                outputs = self(img)
                # Compute loss (Mean Squared Error between output and original image)
                loss = criterion(outputs, img)

                # l1_lambda = 0.0001
                # l1_norm = sum(p.abs().sum() for p in self.parameters())
                # loss += l1_lambda * l1_norm

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                train_loss_items.append(loss.item())

            avg_train_loss = sum(train_loss_items) / len(train_loss_items)
            train_losses.append(avg_train_loss)

            # Validation Phase
            self.eval()
            val_loss_items = []
            with tch.no_grad():
                for val_batch in v_loader:
                    val_img = val_batch['image'].to(self.device)
                    val_outputs = self(val_img)
                    val_loss = criterion(val_outputs, val_img)
                    val_loss_items.append(val_loss.item())

            avg_val_loss = sum(val_loss_items) / len(val_loss_items)
            val_losses.append(avg_val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            scheduler.step()

        return train_losses, val_losses

    def print_eval(self, fold=0):
        self.eval()
        y_true = []
        y_score = []
        # Forward pass through the network and gather predictions and true labels
        with tch.no_grad():
            for batch in self.v_loader:
                data = batch.to(self.device)
                preds = self(data.x, data.edge_index, data.batch)
                # print(f'preds: {preds}')
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
        # print(f'y_score: {y_score}')
        y_pred = [1 if s >= 0.5 else 0 for s in y_score]
        # print(f'y_pred: {y_pred}')
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

    #def weights_init(self):
    #    if isinstance(self, (GCNConv, GATConv)):
    #        nn.init.xavier_uniform_(self.weight.data)
    #    elif isinstance(self, (nn.Linear, nn.Sigmoid)):
    #        nn.init.xavier_uniform_(self.weight.data)
    #    else:
    #        raise ValueError()
        
    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.Module):
            pass  # Do nothing for other types of nn.Module
        else:
            raise ValueError("Unknown layer type for weight initialization")

