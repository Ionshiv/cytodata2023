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

# gpus = tf.config.experimental.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(gpus[0], True)

def main():
    handshake();
    model_type = 'exp147'
    epochs = 3
    model_name = model_type + '_epochs_' + str(epochs)
    class_name = 'CRAE_arch'
    batch_size = 1
    history = runNewModel(model_name=model_name, model_type=model_type, class_name=class_name, epochs=epochs, batch_size=batch_size)
    # predArray, tlen = runTrainedModel();
    # plotPrediction(predArray=predArray)
    # a = np.zeros((25, 100))
    # b = a[:,99]
    # plotAllVectors()
    # plotClusters()
    

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

def runTrainedModel(enIn):
    encoder = load_model(enIn[0])
    encoder.summary()
    encoder.compile(optimizer='Adam', loss='mse')
    # predGen = LcpGenerator(inpath='../data/ki-database/exp147', batch_size = 1, input_aug=False)
    predArray = []
    metaArray = []
    loadpath = sorted(glob.glob(enIn[1]+'/*'))
    tlen = 0
    for i, inpath in enumerate(loadpath):
        print(i)
        print(inpath)
        npseq = np.load(inpath+'/sequence.npy').astype(float)
        npseq = npseq/65535
        tlen = npseq[0]
        npseq = np.reshape(npseq, (1, npseq.shape[0], npseq.shape[1], npseq.shape[2], npseq.shape[3]))
        print(npseq.shape)
        nppred = encoder.predict(npseq)
        predArray += [nppred]
        metaArray += [inpath]
    return predArray, tlen, metaArray

def plotPrediction(predArray, tlen, expname):
    t = range(tlen)
    scatterDim = []
    scatterList = []
    predArray = predArray[:,]
    for j in range(10):
        for i, featMap in enumerate(predArray):
            featMap = np.reshape(featMap, (25, 100))
            feat = featMap[:, j]
            scatterDim += [feat]
        scatterList += [scatterDim[:]]
        scatterDim = []
    fig = plt.figure()
    for k, featVec in enumerate(scatterList):
        print(len(featVec))
        for feat in featVec:
            plt.plot(t, feat, 'o')
        plt.show()
        plot_path = '/scratch2-shared/david/model_data/' + expname
        plot_name =  '/feature_Number_' +str(k) + '.png'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plt.savefig(plot_path + plot_name, format='png')
        plt.close()
   

def plotAllVectors():
    print('')
    en143 = '/scratch-shared/david/model_data/CRAE_arch/exp143_epochs_60_D20211504_T181518encoderSegment', '../data/ki-database/exp143'
    en147 = '/scratch-shared/david/model_data/CRAE_arch/exp147_epochs_60_D20213703_T153737encoderSegment', '../data/ki-database/exp147'
    en156 = '/scratch-shared/david/model_data/CRAE_arch/exp156_epochs_60_D20213908_T233910encoderSegment', '../data/ki-database/exp156'
    pred143, t143 = runTrainedModel(en143)
    plotPrediction(pred143, t143, 'exp143')
    pred147, t147 = runTrainedModel(en147)
    plotPrediction(pred147, t147, 'exp147')
    pred156, t156 = runTrainedModel(en156)
    plotPrediction(pred156, t156, 'exp156')

def plotClusters():
    print('')
    en147 = '/scratch-shared/david/model_data/CRAE_arch/exp147_epochs_60_D20213703_T153737encoderSegment', '../data/ki-database/exp147'
    pred147, t147, meta147 = runTrainedModel(en147)
    print(t147)
    for i in range(24):
        for j in range(100):
            for k in range(100):
                fig = plt.figure(figsize=(15,5), facecolor='w')
                for m, featmap in enumerate(pred147):
                    featmap = np.reshape(featmap, (25, 100))
                    print('')
                    x_val = featmap[i, j]
                    y_val = featmap[i, k]
                    plt.plot(x_val, y_val, '.')
                plt.show()
                plot_path = '/scratch2-shared/david/model_scatterplots/' + 'exp147' +'/feature_X_' +str(j) + '_feature_Y_' + str(k)
                plot_name =  '/timeframe_'+ str(i) + '.png'
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                plt.savefig(plot_path + plot_name, format='png')
                plt.close()


        

    

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
    lcpAutoencoder = LcpAe(25,1080,1080, 6, inpath='/scratch2-shared/david/liveCellPainting/ki-database/exp147', batch_size=batch_size, input_aug=input_aug)
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


