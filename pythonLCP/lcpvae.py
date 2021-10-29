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


class LcpVae:

    def __init__(self):
        self.startstop(True)
        print("not implemented yet")
        self.startstop(False)
    
    def startstop(self, bool_var: bool):

        if bool_var:
            print("+++INNITIALIZING AUTOENCODER+++")
        else:
            print("+++TERMINATING AUTOENCODER+++")
    
    def autoencoder():
        # encoder = layers.sequential()
        # decoder = layers.sequential()

        ingress = layers.Input(1080, 1080, 5)
        encoder = layers.Conv2D(padding = 'same', kernel_size=(3,3), activation='relu')(ingress)
        encoder = layers.Conv2D(padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.MaxPool2D(pool_size=(2, 2), strides=2)(encoder)
        encoder = layers.Conv2D(padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)
        encoder = layers.Conv2D(padding='same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(padding='same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)
        encoder = layers.Conv2D(padding='same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(padding='same', kernel_size=(3,3), activation='relu')(encoder)

        transcoder = layers.Flatten()(encoder)
        transcoder = Dense(100, activation='softmax')(transcoder)
        transcoder = Dense(20, activation='softmax')(transcoder)
        recursion = layers.SimpleRNN(2)(transcoder)
        transcoder = Dense(20, activation='softmax')(recursion)
        transcoder = Dense(100, activation='softmax')(transcoder)
        
        decoder = Reshape((145, 145, 5))(transcoder)
        decoder = layers.Conv2DTranspose(padding='same', kernel_size=(3,3), activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(padding='same', kernel_size=(3,3), activation='relu')(decoder)
        decoder = layers.UpSampling2D(size=(2, 2))(decoder)
        decoder = layers.Conv2DTranspose(padding='same', kernel_size=(3,3), activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(padding='same', kernel_size=(3,3), activation='relu')(decoder)
        decoder = layers.UpSampling2D(size=(2, 2))(decoder)
        decoder = layers.Conv2DTranspose(padding='same', kernel_size=(3,3), activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(padding='same', kernel_size=(3,3), activation='relu')(decoder)
        decoder = layers.UpSampling2D(size=(2, 2))(decoder)
        decoder = layers.Conv2DTranspose(padding='same', kernel_size=(3,3), activation='relu')(decoder)
        egress = layers.Conv2DTranspose(padding='same', kernel_size=(3,3), activation='relu')(decoder)

        aemodel = Model(ingress, egress)




        # encoder.add(layers.Conv2D(4,
        #     padding = 'same',
        #     kernel_size=(3,3),
        #     input_shape=(1080, 1080, 6),
        #     activation = 'relu'))
        # encoder.add(layers.Conv2D(4,
        #     padding = 'same',
        #     kernel_size=(3,3),
        #     input_shape=(1080, 1080, 6),
        #     activation = 'relu'))
        # encoder.add(layers.Conv2D(4,
        #     padding = 'same',
        #     kernel_size=(3,3),
        #     input_shape=(1080, 1080, 6),
        #     activation = 'relu'))
        # encoder.add(layers.MaxPool2D(pool_size=(2, 2)))

        print("soon")
        
  