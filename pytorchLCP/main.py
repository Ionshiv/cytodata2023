import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn
import torch as torch
import torchvision as v
import torch.nn.functional as nnf
import os as os
from datetime import datetime

def main():
    print('torchswitch')


def plot_history(history, model_name, class_name):
    fig = plt.figure(figsize=(15,5), facecolor='w')
    ax = fig.add_subplot(121)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set(title=model_name + ': Model loss', ylabel='Loss', xlabel='Epoch')
    ax.legend(['Train', 'Test'], loc='upper right')
    ax = fig.add_subplot(122)
    ax.plot(np.log(history.history['loss']))
    ax.plot(np.log(history.history['val_loss']))
    ax.set(title=model_name + ': Log model loss', ylabel='Log loss', xlabel='Epoch')
    ax.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    plot_path = '../model_data/' + class_name
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    timestampObj = datetime.now()
    timestampStr = timestampObj.strftime("_D%Y%M%d_T%H%M%S")
    plt.savefig(plot_path+'/'+ model_name +timestampStr + '.png', format='png')
    plt.close()

if __name__=="__main__":
    main();
