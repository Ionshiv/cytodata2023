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
        self.startstop(True)
        print("not implemented yet")
        # self.testLayers()
        self.autoencoder()
        self.startstop(False)
    
    def startstop(self, bool_var: bool):

        if bool_var:
            print("+++INNITIALIZING AUTOENCODER+++")
        else:
            print("+++TERMINATING AUTOENCODER+++")
    
    def autoencoder(self):
        # encoder = layers.sequential()
        # decoder = layers.sequential()


        ingress = layers.Input((24, 1080, 1080, 6))
        encoder = layers.Conv2D(filters=6, padding = 'same', kernel_size=(3,3), activation='relu')(ingress)
        encoder = layers.Conv2D(filters=6, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown0 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(encoder)
        print('Encoder layer 0 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown0)
        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown1 = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)
        print('Encoder layer 1 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown1)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown2 = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)
        print('Encoder layer 2 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown2)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown3 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)
        print('Encoder layer 3 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown3)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown4 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)
        print('Encoder layer 4 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown4)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown5 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)
        print('Encoder layer 5 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown5)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown6 = layers.Flatten()(encoder)
        num_el = encoderDown6.shape[1]
        print('Encoder layer 6 shape: ', encoder.shape)
        
        # transcoder = layers.Dense(encoderDown6.shape[1], activation='relu')(encoderDown6)
        # print('transcoder in shape: ',transcoder.shape)
        # transcoder = layers.Dense(500, activation='relu')(transcoder)
        # transcoder = layers.Dense(100, activation='relu')(transcoder)
        # transcoder = layers.Dense(20, activation='relu')(transcoder)
        # print('transcoder bottleneck shape: ',transcoder.shape)
        # transcoder = layers.Dense(100, activation='relu')(transcoder)
        # transcoder = layers.Dense(500, activation='relu')(transcoder)
        # transcoder = layers.Dense(num_el, activation='relu')(transcoder)

        bottleneck = layers.GRU(num_el, activation='relu')(encoderDown6)
        bottleneck = layers.GRU(500, activation='relu')(bottleneck)
        bottleneck = layers.GRU(100, activation='relu')(bottleneck)
        bottleneck = layers.GRU(20, activation='relu')(bottleneck)
        bottleneck = layers.GRU(100, activation='relu')(bottleneck)
        bottleneck = layers.GRU(500, activation='relu')(bottleneck)

        print('transcoder out shape: ',bottleneck.shape)
        
        decoderUp0 = layers.Reshape((encoder.shape[1], encoder.shape[2], encoder.shape[3]))(bottleneck)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoderUp0)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 0 shape: ', decoder.shape)

        decoderUp1 = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoderUp1)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 1 shape: ', decoder.shape)

        decoderUp2 = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoderUp2)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 2 shape: ', decoder.shape)

        decoderUp3 = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoderUp3)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 3 shape: ', decoder.shape)

        decoderUp4 = layers.Conv2DTranspose(filters=18, kernel_size=(2,2), strides=(2,2), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoderUp4)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 4 shape: ', decoder.shape)

        decoderUp5 = layers.Conv2DTranspose(filters=12, kernel_size=(2,2), strides=(2,2), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoderUp5)
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 5 shape: ', decoder.shape)

        decoderUp6 = layers.Conv2DTranspose(filters=6, kernel_size=(2,2), strides=(2,2), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), padding='same', activation='relu')(decoderUp6)
        decoder = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        egress = layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid')(decoder)
        print('Decoder layer 6 shape: ', decoder.shape)
        print('Egress layer shape: ', egress.shape)
        print("soon(TM)")

    def testLayers(self):

        tin = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(tin)
        print(tin.shape[0])
        print(tin.shape[1])
        x = layers.Input((tin.shape[0], tin.shape[1], 1))
        
        print(x)
        # x = layers.UpSampling2D(size=(2,2))(x)
        # print(x)
        # x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
        # print(x)
        x = layers.Conv2DTranspose(filters=3, strides=(3,3), padding='same', kernel_size=(3,3), activation='relu')(x)
        print(x)
        # x = layers.Conv2DTranspose(filters=3, strides=(1,1), padding='same', kernel_size=(3,3), activation='relu')(x)
        x = layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu' )(x)
        print(x)
        # x = layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))(x)
        # print(x.shape)
        # print(x)

        
  