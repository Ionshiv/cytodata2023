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
        self.crAutoencoder = self.autoencoder();
        self.latentOutputLayer = self.crAutoencoder.get_layer(name='LatentLayer');
        self.startstop(False);

    
    def startstop(self, bool_var: bool):
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

        ingress = layers.Input((1080, 1080, 6))
        encoder = layers.Conv2D(filters=6, padding = 'same', kernel_size=(3,3), activation='relu')(ingress)
        encoder = layers.Conv2D(filters=6, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown0 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(encoder)

        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown0)
        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown1 = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)

        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown1)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown2 = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)

        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown2)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown3 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)

        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown3)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown4 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)

        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown4)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown5 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)

        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown5)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown6 = layers.Flatten()(encoder)
        subEncoder = self.subEncoder = Model(ingress, encoderDown6)
        rEnIn = layers.Input((24, 1080, 1080, 6))
        rEncoderSegment = layers.TimeDistributed(subEncoder, name='timeIn')(rEnIn)


        bottleneck = bIn = layers.GRU(500, return_sequences=True)(rEncoderSegment)
        bottleneck = layers.GRU(100, return_sequences=True)(bottleneck)
        bottleneck = latentLayer = layers.GRU(20, return_sequences=True, name='LatentLayer')(bottleneck)
        bottleneck = layers.GRU(100, return_sequences=True)(bottleneck)
        bottleneck = bOut = layers.GRU(500, return_sequences=True)(bottleneck)


        dIngress = layers.Input(bOut.shape[2])
        dFormat = layers.Dense(encoderDown6.shape[1], activation='relu')(dIngress)
        decoderUp0 = layers.Reshape((encoder.shape[1], encoder.shape[2], encoder.shape[3]))(dFormat)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoderUp0)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)

        decoderUp1 = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoderUp1)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)

        decoderUp2 = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoderUp2)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)

        decoderUp3 = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), strides=(3,3), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoderUp3);
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder);

        decoderUp4 = layers.Conv2DTranspose(filters=18, kernel_size=(2,2), strides=(2,2), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoderUp4);
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder);

        decoderUp5 = layers.Conv2DTranspose(filters=12, kernel_size=(2,2), strides=(2,2), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoderUp5);
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoder);

        decoderUp6 = layers.Conv2DTranspose(filters=6, kernel_size=(2,2), strides=(2,2), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), padding='same', activation='relu')(decoderUp6);
        decoder = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        egress = layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid')(decoder);
        print('Egress layer shape: ', egress.shape);

        subDecoder = self.subDecoder = Model(dIngress, egress);
        rDecoderSegment = layers.TimeDistributed(subDecoder)(bOut);

        fullAutoencoder = Model(rEnIn, rDecoderSegment);
        return fullAutoencoder;


    def testLayers(self):
        print('testing GLOBAL Autoencoder storage: ',self.rAE.get_layer);
        print('Testing GLOBAL latent Layer transmission: ',self.ll.output);
        print('Testing layer shape: ', self.ll.output.shape);

        



        
  