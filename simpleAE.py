from __future__ import absolute_import, division, print_function, unicode_literals

# Pytorch
import torch as tch
import torch.nn as nn
import torch.optim as optim

# External Helper libraries
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


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        
        layers = [
            nn.Conv2d(input_dim, 9, kernel_size=3, stride=2, padding=1),  # 512x512x9
            nn.Conv2d(9, 12, kernel_size=3, stride=2, padding=1),  # 256x256x12
            nn.Conv2d(12, 15, kernel_size=3, stride=2, padding=1),  # 128x128x15
            nn.Conv2d(15, 18, kernel_size=3, stride=2, padding=1),  # 64x64x18
            nn.Conv2d(18, 21, kernel_size=3, stride=2, padding=1),  # 32x32x21
            nn.Conv2d(21, 24, kernel_size=3, stride=2, padding=1),  # 16x16x24
            nn.Conv2d(24, 27, kernel_size=3, stride=2, padding=1),  # 8x8x27
            nn.Conv2d(27, 30, kernel_size=3, stride=2, padding=1),  # 4x4x30
            nn.Conv2d(30, 33, kernel_size=3, stride=2, padding=1),  # 2x2x33
        ]
        self.enc_convs = nn.ModuleList(layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs = []
        for enc_conv in self.enc_convs:
            x = self.relu(enc_conv(x))
            outputs.append(x)
        return outputs

class Decoder(nn.Module):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        
        layers = [
            nn.ConvTranspose2d(33, 30, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(30, 27, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(27, 24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(24, 21, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(21, 18, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(18, 15, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(15, 12, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(12, 9, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(9, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        ]
        self.dec_convs = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, enc_outputs):
        for i, dec_conv in enumerate(self.dec_convs):
            x = self.relu(dec_conv(x))
            # x = self.relu(dec_conv(x) + enc_outputs[-(i)])
        return self.sigmoid(x)

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
        self.encoder = Encoder(6)
        self.decoder = Decoder(6)

    def forward(self, x):
        enc_outputs = self.encoder(x)
        dec_output = self.decoder(enc_outputs[-1], enc_outputs[:-1])
        return dec_output
  


    def fitAE(self, t_loader, v_loader, num_epochs, optimizer, criterion, scheduler):
        self.train()
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            # Training Phase
            self.train()
            train_loss_items = []
            for batch in t_loader:
                img = batch.to(self.device)
                img = tch.squeeze(img)
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
                    val_img = val_batch.to(self.device)
                    val_img = tch.squeeze(val_img)
                    val_outputs = self(val_img)
                    val_loss = criterion(val_outputs, val_img)
                    val_loss_items.append(val_loss.item())

            avg_val_loss = sum(val_loss_items) / len(val_loss_items)
            val_losses.append(avg_val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            scheduler.step()
        self.t_loader = t_loader
        self.v_loader = v_loader
        return train_losses, val_losses

    def make_embedding(self, data_loader=None):
        self.encoder.eval()  # Set the encoder to evaluation mode
        embeddings = []
        if data_loader:
            print('full_loader')
            lname = 'external_'
        else:
            data_loader = self.t_loader
            lname = 'training_data_'
        with tch.no_grad():  # No need to calculate gradients
            for batch in data_loader:
                data = batch.to(self.device)
                data = tch.squeeze(data)
                enc_outputs = self.encoder(data)
                # Take just the last output tensor from the list of encoder outputs
                final_output = enc_outputs[-1].cpu().numpy()
                final_output = final_output.reshape(-1)
                embeddings.append(final_output)

        embeddings = np.vstack(embeddings)
        np.save(f'{lname}encoder_embeddings.npy', embeddings)
        return embeddings

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

