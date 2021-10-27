from keras.engine.sequential import Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import skimage as ski
from tensorflow import keras
from keras import layers


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
    
    def reticulum():
        # encoder = layers.sequential()
        # decoder = layers.sequential()

        inputs = layers.Input(1080, 1080, 5)
        encoder = layers.Conv2D(padding = 'same', kernel_size=(3,3), activation='relu')(inputs)
        encoder = layers.Conv2D(padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.MaxPool2D(pool_size=(2, 2))(encoder)
        encoder = layers.Conv2D(padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(padding = 'same', kernel_size=(3,3), activation='relu')(encoder)

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
        
  