from keras.engine.sequential import Sequential
from keras.layers.core import Dense, Reshape
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import skimage as ski
import PIL as pil
from tensorflow import keras
from keras import layers
from keras.models import Model


class LcpAe:

    def __init__(self):
        self.startstop(True);
        self.startstop(False);

    def seqGenerator(self, vidpath:str):
        print('soon(TM)');

    def buildAutoEncoder(self, timeframes:int, m:int, n:int, channels:int):
        self.timeframes = timeframes;
        self.m = m;
        self.n = n;
        self.channels=channels;
        self.crAutoencoder = self.autoencoder();
        self.latentOutputLayer = self.crAutoencoder.get_layer(name='LatentLayer');
        print('soon(TM');

    def startstop(self, bool_var:bool):
        if bool_var:
            print("+++INNITIALIZING AUTOENCODER CONSTRUCTOR+++");
        else:
            print("+++TERMINATING AUTOENCODER CONSTRUCTOR+++");
    
    def getAutoencoderModel(self):
        return self.crAutoencoder;
    
    def getLatentSpaceLayer(self):
        return self.latentOutputLayer;

    def compileAutoencoder(self):
        self.crAutoencoder.compile(optimizer='adam', loss='mse');

    def getSummaryAutoencoder(self):
        self.subEncoder.summary();
        self.crAutoencoder.summary();
        self.subDecoder.summary();
        

    def autoencoder(self):
        ingress = layers.Input((self.m, self.n, self.channels))
        encoder = layers.Conv2D(filters=self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(ingress)
        encoder = layers.Conv2D(filters=self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown0 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(encoder)

        encoder = layers.Conv2D(filters=2*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown0)
        encoder = layers.Conv2D(filters=2*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=2*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown1 = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)

        encoder = layers.Conv2D(filters=3*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown1)
        encoder = layers.Conv2D(filters=3*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=3*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown2 = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)

        encoder = layers.Conv2D(filters=4*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown2)
        encoder = layers.Conv2D(filters=4*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=4*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown3 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)

        encoder = layers.Conv2D(filters=5*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown3)
        encoder = layers.Conv2D(filters=5*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=5*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown4 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)

        encoder = layers.Conv2D(filters=6*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown4)
        encoder = layers.Conv2D(filters=6*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=6*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown5 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)

        encoder = layers.Conv2D(filters=7*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown5)
        encoder = layers.Conv2D(filters=7*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=7*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown6 = layers.Flatten()(encoder)
        subEncoder = self.subEncoder = Model(ingress, encoderDown6)
        rEnIn = layers.Input((self.timeframes, self.m, self.n, self.channels))
        rEncoderSegment = layers.TimeDistributed(subEncoder, name='timeIn')(rEnIn)


        bottleneck = bIn = layers.GRU(500, return_sequences=True)(rEncoderSegment)
        bottleneck = layers.GRU(100, return_sequences=True)(bottleneck)
        bottleneck = latentLayer = layers.GRU(20, return_sequences=True, name='LatentLayer')(bottleneck)
        bottleneck = layers.GRU(100, return_sequences=True)(bottleneck)
        bottleneck = bOut = layers.GRU(500, return_sequences=True)(bottleneck)


        dIngress = layers.Input(bOut.shape[2])
        dFormat = layers.Dense(encoderDown6.shape[1], activation='relu')(dIngress)
        decoderUp0 = layers.Reshape((encoder.shape[1], encoder.shape[2], encoder.shape[3]))(dFormat)
        decoder = layers.Conv2DTranspose(filters=7*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp0)
        decoder = layers.Conv2DTranspose(filters=7*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=7*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder)

        decoderUp1 = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp1)
        decoder = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder)

        decoderUp2 = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp2)
        decoder = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder)

        decoderUp3 = layers.Conv2DTranspose(filters=4*self.channels, kernel_size=(3,3), strides=(3,3), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=4*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp3);
        decoder = layers.Conv2DTranspose(filters=4*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=4*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);

        decoderUp4 = layers.Conv2DTranspose(filters=3*self.channels, kernel_size=(2,2), strides=(2,2), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=3*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp4);
        decoder = layers.Conv2DTranspose(filters=3*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=3*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);

        decoderUp5 = layers.Conv2DTranspose(filters=2*self.channels, kernel_size=(2,2), strides=(2,2), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=2*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp5);
        decoder = layers.Conv2DTranspose(filters=2*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=2*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);

        decoderUp6 = layers.Conv2DTranspose(filters=self.channels, kernel_size=(2,2), strides=(2,2), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp6);
        decoder = layers.Conv2DTranspose(filters=self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        egress = layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid')(decoder);
        print('Egress layer shape: ', egress.shape);

        subDecoder = self.subDecoder = Model(dIngress, egress);
        rDecoderSegment = layers.TimeDistributed(subDecoder)(bOut);

        fullAutoencoder = Model(rEnIn, rDecoderSegment);
        return fullAutoencoder;


    def testLayers(self):
        print('Dormant')

        



        
  