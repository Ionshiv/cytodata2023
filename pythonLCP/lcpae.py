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
        self.testLayers()
        # self.autoencoder()
        self.startstop(False)
    
    def startstop(self, bool_var: bool):

        if bool_var:
            print("+++INNITIALIZING AUTOENCODER+++")
        else:
            print("+++TERMINATING AUTOENCODER+++")
    
    def autoencoder(self):
        # encoder = layers.sequential()
        # decoder = layers.sequential()

        tingress = layers.Input((24, 1080, 1080, 6))


        # ingress = layers.Input((24, 1080, 1080, 6))
        ingress = layers.Conv2D(filters=6, padding='same', kernel_size=(1,1), activation='none')
        encoder = layers.Conv2D(filters=6, padding = 'same', kernel_size=(3,3), activation='relu')(ingress)
        encoder = layers.Conv2D(filters=6, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown0 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(encoder)
        print('Encoder layer 0 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown0)
        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=12, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown1 = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)
        print('Encoder layer 1 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown1)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=18, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown2 = layers.MaxPool2D(pool_size=(2,2), strides=2)(encoder)
        print('Encoder layer 2 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown2)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=24, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown3 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)
        print('Encoder layer 3 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown3)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=30, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown4 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)
        print('Encoder layer 4 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown4)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=36, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown5 = layers.MaxPool2D(pool_size=(3,3), strides=3)(encoder)
        print('Encoder layer 5 shape: ', encoder.shape)

        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown5)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoder = layers.Conv2D(filters=42, padding = 'same', kernel_size=(3,3), activation='relu')(encoder)
        encoderDown6 = layers.Flatten()(encoder)
        num_el = encoderDown6.shape[1]
        print('Encoder layer 6 shape: ', encoder.shape)

        # cIngress = layers.Input(encoderDown6.shape[1])
        cIngress = layers.GRU(num_el, activation='relu')
        compressor = layers.GRU(500, activation='relu')(cIngress)
        compressor = layers.GRU(100, activation='relu')(compressor)
        latentLayer = layers.GRU(20, activation='relu')(compressor)
        print('Latent vector space: ', latentLayer.shape)


        # eIngress = layers.Input(latentLayer.shape[1])
        eIngress = layers.GRU(100, activation='relu')
        eEgress = layers.GRU(500, activation='relu')(eIngress)
        print('Expander out shape: ',eEgress.shape)


        dIngress = layers.Input((encoder.shape[1], encoder.shape[2], encoder.shape[3]))
        decoderUp0 = layers.Reshape((encoder.shape[1], encoder.shape[2], encoder.shape[3]))(dIngress)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoderUp0)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=42, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 0 shape: ', decoder.shape)

        decoderUp1 = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoderUp1)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=36, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 1 shape: ', decoder.shape)

        decoderUp2 = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoderUp2)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=30, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 2 shape: ', decoder.shape)

        decoderUp3 = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), strides=(3,3), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoderUp3)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=24, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 3 shape: ', decoder.shape)

        decoderUp4 = layers.Conv2DTranspose(filters=18, kernel_size=(2,2), strides=(2,2), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoderUp4)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=18, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 4 shape: ', decoder.shape)

        decoderUp5 = layers.Conv2DTranspose(filters=12, kernel_size=(2,2), strides=(2,2), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoderUp5)
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        decoder = layers.Conv2DTranspose(filters=12, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        print('Decoder layer 5 shape: ', decoder.shape)

        decoderUp6 = layers.Conv2DTranspose(filters=6, kernel_size=(2,2), strides=(2,2), padding='same')(decoder)
        decoder = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), padding='same', activation='relu')(decoderUp6)
        decoder = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), padding='same', activation='relu')(decoder)
        egress = layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid')(decoder)
        print('Decoder layer 6 shape: ', decoder.shape)
        print('Egress layer shape: ', egress.shape)


        encoderSegment = Model(inputs=ingress, outputs=encoderDown6)
        compressorSegment = Model(inputs=cIngress, outputs=latentLayer)
        expanderSegment = Model(inputs=eIngress, outputs=eEgress)
        decoderSegment = Model(inputs=dIngress, outputs=egress)
        
        rEncoder = layers.TimeDistributed(encoderSegment)(compressorSegment)
        rDecoder = layers.TimeDistributed(expanderSegment)(decoderSegment)




        print("soon(TM)")

    def testLayers(self):

        # tin = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # print(tin)
        # print(tin.shape[0])
        # print(tin.shape[1])
        # x = layers.Input((tin.shape[0], tin.shape[1], 1))
        
        # print(x)
        # x = layers.UpSampling2D(size=(2,2))(x)
        # print(x)
        # x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
        # print(x)
        # x = layers.Conv2DTranspose(filters=3, strides=(3,3), padding='same', kernel_size=(3,3), activation='relu')(x)
        # print(x)
        # x = layers.Conv2DTranspose(filters=3, strides=(1,1), padding='same', kernel_size=(3,3), activation='relu')(x)
        # x = layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu' )(x)
        # print(x)
        # x = layers.MaxPool2D(pool_size=(2,2), strides=(2, 2))(x)
        # print(x.shape)
        # print(x)


        #testing time arrays
        tin = np.zeros((9, 6, 6))
        print(tin.shape)
        print(tin[0])
        for i in range(9):
            # tin[i, :, :] = np.array([[i, i, i], [i, 0, i], [i, i, i]])
            tin[i, :, :] = np.array([[i, i, i, i, i, i],[i, 0, 0, 0, 0, i],[i, 0, i, i, 0, i],[i,0,i,i,0,i],[i,0,0,0,0,i],[i,i,i,i,i,i]])
            # print('test')
            # print(tin[i])
        # x = layers.Conv2D(3, kernel_size=(1,1), padding='same', activation=None)
        x = xin = layers.Input((6, 6, 1))
        x = layers.Conv2D(3, kernel_size=(3,3), padding='same', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
        x = xout = layers.Flatten()(x)
        xmodel = Model(xin, xout)
        rxin = layers.Input((9,6,6,1))
        rx = layers.TimeDistributed(xmodel)(rxin)



        # y = yin = layers.Input((9, x.shape[1]))
        y = yin = layers.GRU(rx.shape[1], return_sequences=True)(rx)
        y = latent_layer = layers.GRU(3, return_sequences=True)(y)
        y = yout = layers.GRU(x.shape[1], return_sequences=True)(y)
        print('RNN segment out: ', yout.shape)

        z = zin = layers.Input(yout.shape[2])
        print("zin: ", zin.shape)
        z = layers.Reshape((3, 3, 3))(z)
        z = layers.Conv2DTranspose(3,  kernel_size=(2,2), strides=(2,2), padding='same', activation='relu')(z)
        z = zout = layers.Conv2DTranspose(3, kernel_size=(3, 3), padding='same', activation='relu')(z)


        # xmodel = Model(inputs=xin, outputs=xout)
        # ymodel = Model(yin, yout)
        # print('ymodel type: ', type(ymodel) )
        zmodel = Model(zin, zout)

        # rxin = layers.Input((9, 6, 6, 1))
        rzin = layers.Input((9, 27, 1))
        rz = layers.TimeDistributed(zmodel)(rzin)

        Autoencoder = Model(rxin, rz)

        # rx_wrapper = layers.TimeDistributed(xmodel)(rxin)
        # print('rx_wrapper: ', type(rx_wrapper))
        # ryx_layer = ymodel(rx_wrapper)
        # print('ryx_layer: ', type(ryx_layer))
        # ryxmodel = Model(rxin, ryx_layer)
        # rz_wrapper= layers.TimeDistributed(zmodel)(rzin)
        # print('ry_wrapper: ', type(rz_wrapper))
        # # rzx_layer = ryxmodel(rz_wrapper)
        # rzyxmodel = Model(xin, rz_wrapper)
        # # rzxmodel = Model(rxin, rz_wrapper)
        # # rzxmodel = ry_wrapper(rxmodel)
        # print('rzyxmodel: ', rzyxmodel.shape)



        
  