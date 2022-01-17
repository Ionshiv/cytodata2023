import numpy as np
import os as os
from datetime import datetime
from tensorflow.python.ops.control_flow_ops import case
from lcpae import LcpAe
from lcpgenerator import LcpGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import glob as glob


def main():
    handshake();
    model_type = 'exp156'
    epochs = 60
    model_name = model_type + '_epochs_' + str(epochs)
    class_name = 'CRAE_arch'
    batch_size = 1
    # history = runNewModel(model_name=model_name, model_type=model_type, class_name=class_name, epochs=epochs, batch_size=batch_size)
    predArray = runTrainedModel();
    plotPrediction(predArray=predArray)
    # a = np.zeros((25, 100))
    # b = a[:,99]
    

    endshake();



def handshake():
    print("+++INNITIALIZING SESSION+++");

def endshake():
    print("+++ENDING SESSION+++");

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

def buildAutoencoder(case:str, batch_size:int, input_aug:bool=False):
    
    lcpAutoencoder = exp_case(case, batch_size, input_aug=input_aug);
    return lcpAutoencoder

def runNewModel(model_type:str, model_name:str, class_name:str, epochs:int, batch_size:int):
    lcpAutoencoder = buildAutoencoder(case=model_type, batch_size=batch_size);
    lcpAutoencoder.compileAutoencoder();
    lcpAutoencoder.getSummaryAutoencoder();
    # lcpAutoencoder.buildSegmentedAutoencoder();
    history = lcpAutoencoder.fitAutoencoder(epochs=epochs);
    plot_history(history, model_name, class_name)
    if not os.path.exists('../model_data/'+class_name):
        os.makedirs('../model_data/'+class_name)
    timestampObj = datetime.now()
    timestampStr = timestampObj.strftime("_D%Y%M%d_T%H%M%S")
    lcpAutoencoder.autoencoder.save('../model_data/'+class_name+'/'+model_name+timestampStr+'full')
    lcpAutoencoder.encoder.save('../model_data/'+class_name+'/'+model_name+timestampStr+'encoderSegment')
    # lcpAutoencoder.encoder.predict()
    return history

def runTrainedModel():
    encoder = load_model('/scratch-shared/david/model_data/CRAE_arch/exp147_epochs_60_D20213703_T153737encoderSegment')
    encoder.summary()
    encoder.compile(optimizer='Adam', loss='mse')
    # predGen = LcpGenerator(inpath='../data/ki-database/exp147', batch_size = 1, input_aug=False)
    predArray = []
    loadpath = sorted(glob.glob('../data/ki-database/exp147'+'/*'))
    loadpath = loadpath[3::4]
    for i, inpath in enumerate(loadpath):
        print(i)
        print(inpath)
        npseq = np.load(inpath+'/sequence.npy').astype(float)
        npseq = npseq/65535
        npseq = np.reshape(npseq, (1, npseq.shape[0], npseq.shape[1], npseq.shape[2], npseq.shape[3]))
        print(npseq.shape)
        nppred = encoder.predict(npseq)
        predArray += [nppred]
    return predArray

def plotPrediction(predArray):
        print('')
        index = 0
        for nppred in predArray:
            index += 1
            nppred = np.reshape(nppred, (25, 100))
            # print(nppred.shape)
            # plt.figure(figsize=(10, 10), facecolor='w')
            # plt.imshow(nppred, cmap='hot')
            # plt.show()
            # plt.savefig('../data/fig'+str(index)+'147.png')
            # plt.close()
            
        

    

def exp_case(case, batch_size, input_aug:bool=False):
    # lcpAutoencoder = LcpAe();
    switcher = {
        'exp143': exp143ae,
        'exp147': exp147ae,
        'exp156': exp156ae,
        'exp180': exp180ae
    }
    lcpAutoencoder = switcher[case](batch_size, input_aug)
    return lcpAutoencoder;

def exp143ae(batch_size:int, input_aug:bool=False):
    print('+++ GENERATING: exp143 +++')
    # lcpGen = LcpGenerator(inpath='../data/ki-database/exp143', batch_size=batch_size)
    lcpAutoencoder = LcpAe(24,1080,1080, 6, inpath='../data/ki-database/exp143', batch_size=batch_size, input_aug=input_aug)
    return lcpAutoencoder

def exp147ae(batch_size:int, input_aug:bool=False):
    print('+++ GENERATING: exp147 +++')
    # lcpGen = LcpGenerator(inpath='../data/ki-database/exp147', batch_size=batch_size, input_aug=True)
    lcpAutoencoder = LcpAe(25,1080,1080, 6, inpath='../data/ki-database/exp147', batch_size=batch_size, input_aug=input_aug)
    return lcpAutoencoder

def exp156ae(batch_size:int, input_aug:bool=False):
    print('+++ GENERATING: exp156 +++')
    # lcpGen = LcpGenerator(inpath='../data/ki-database/exp156', batch_size=batch_size)
    lcpAutoencoder = LcpAe(14,1080,1080, 5, inpath='../data/ki-database/exp156', batch_size=batch_size, input_aug=input_aug)
    return lcpAutoencoder

def exp180ae(batch_size:int, input_aug:bool=False):
    print('+++ GENERATING: exp180 +++')
    # lcpGen = LcpGenerator(inpath='../data/ki-database/exp180', batch_size=batch_size)
    lcpAutoencoder = LcpAe(24,1080,1080, 6, inpath='../data/ki-database/exp180', batch_size=batch_size, input_aug=input_aug)
    return lcpAutoencoder


if __name__ == "__main__":

    main();

