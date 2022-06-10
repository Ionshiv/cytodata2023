from configparser import ConverterMapping
import numpy as np
import os as os
import sklearn as sk
import glob as glob
from PIL import Image
import plotly.express as px
import sklearn as sk
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class LcpPca:

    def __init__(self):
        self.impath = sorted(glob.glob('/scratch2-shared/david/liveCellPainting/ki-database/exp180/*'))
        self.pcapath = '/scratch2-shared/david/liveCellPainting/model_data/PCA/exp180/'
        self.runnum = 'run1'
        if not os.path.exists(self.pcapath+self.runnum):
            os.makedirs(self.pcapath+self.runnum)
        self.startPCA()

    def startPCA(self):
        convdata = np.zeros((25,2,6))
        
        pca = PCA(2)
        for wellpath in self.impath:
            print(wellpath)
            _ , tailbit = os.path.split(wellpath)
            print(tailbit)
            for i in range(6):
                imgin = np.load(wellpath+'/ch'+str(i)+'.npy')
                print(imgin.shape)
                # t0=imgin[0,:,:]
                # print(t0.shape)
                # t0 = np.reshape(t0, (1, 1166400))
                # print(t0.shape)
                # t24 = imgin[24,:,:]
                # t24 = np.reshape(t24, (1, 1166400))
                
                # convt0 = pca.fit_transform(t0)
                # convt24 = pca.fit_transform(t24)
                # print(convt0.shape)
                imgin = np.reshape(imgin, (25, 1166400))
                convim = pca.fit_transform(imgin)
                print(convim.shape)
                convdata[:,:,i]=convim
            np.save(self.pcapath+self.runnum+'/'+tailbit+'.npy', convdata)



            
                
            
            
        
