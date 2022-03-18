from keras.engine import input_layer
import tensorflow as tf
import numpy as np
import sklearn as sk
from tensorflow import keras
from keras import layers
from keras.models import Model
from lcpgenerator import LcpGenerator


class LcpAe:

    def __init__(self, timeframes:int, m:int, n:int, channels:int, inpath:str, batch_size:int=1, input_aug:bool=False):
        self.startstop(True);
        self.inpath = inpath;
        self.batch_size = batch_size;
        self.input_aug = input_aug;
        self.timeframes = timeframes;
        self.m = m;
        self.n = n;
        self.channels=channels;
        self.startGenerator();
        # self.testGenerator();
        self.autoencoder, self.encoder, self.decoder = self.buildFullAutoencoder();
        # self.autoencoder = self.buildTest();
        self.startstop(False);

    def startGenerator(self):
        self.lcpGen = LcpGenerator(inpath=self.inpath, batch_size=self.batch_size, input_aug=self.input_aug);

    def testGenerator(self):
        for i in range(3):
            A, B = next(self.lcpGen.trainGen());
            print(A.shape, B.shape)

    def startstop(self, bool_var:bool):
        if bool_var:
            print("+++INNITIALIZING AUTOENCODER CONSTRUCTOR+++");
        else:
            print("+++TERMINATING AUTOENCODER CONSTRUCTOR+++");
    
    def getAutoencoderModel(self):
        return self.autoencoder;

    def compileAutoencoder(self):
        # self.encoder.compile(optimizer='adam', loss='binary_crossentropy');
        # self.decoder.compile(optimizer='adam', loss='binary_crossentropy');
        optim = keras.optimizers.Adam(learning_rate=0.0002)
        self.autoencoder.compile(optimizer=optim, loss='mse');

    def getSummaryAutoencoder(self):
        self.x_model.summary();
        self.autoencoder.summary();
        self.y_model.summary();

    def getSummaryExtractor(self):
        self.encoder.summary();
    
    def fitAutoencoder(self, epochs:int):
        aug_factor = 1
        if self.input_aug:
            aug_factor = 4;
        history = self.autoencoder.fit(x=self.lcpGen.trainGen(), epochs=epochs, batch_size=aug_factor*self.batch_size, steps_per_epoch=aug_factor*len(self.lcpGen.trainpath)//self.batch_size, verbose=1, validation_data=self.lcpGen.validGen(), validation_steps=aug_factor*len(self.lcpGen.validpath)//self.batch_size)
        return history;

    def buildFullAutoencoder(self):
        autoencoder, encoder, decoder = self.buildNetworks();
        return autoencoder, encoder, decoder

    def buildNetworks(self):
        ingress = layers.Input((self.m, self.n, self.channels))
        x = layers.Conv2D(filters=self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(ingress)
        encoderDown0 = layers.Conv2D(filters=1*self.channels, padding='same', kernel_size=(3, 3), strides=(2,2), activation='relu')(x)
        #540
        x = layers.Conv2D(filters=2*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown0)
        encoderDown1 = layers.Conv2D(filters=3*self.channels, padding='same', kernel_size=(5, 5), strides=(2,2), activation='relu')(x)
        #270
        x = layers.Conv2D(filters=3*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown1)
        encoderDown2 = layers.Conv2D(filters=4*self.channels, padding='same', kernel_size=(5, 5), strides=(2,2), activation='relu')(x)
        #135 - HERE
        x = layers.Conv2D(filters=4*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown2)
        encoderDown3 = layers.Conv2D(filters=5*self.channels, padding='same', kernel_size=(5, 5), strides=(3,3), activation='relu')(x)
        #45
        x = layers.Conv2D(filters=5*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown3)
        encoderDown4 = layers.Conv2D(filters=6*self.channels, padding='same', kernel_size=(5, 5), strides=(3,3), activation='relu')(x)
        #15 - HERE
        x = layers.Conv2D(filters=6*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown4)
        encoderDown5 = layers.Conv2D(filters=7*self.channels, padding='same', kernel_size=(5, 5), strides=(3,3), activation='relu')(x)
        #5
        x = layers.Conv2D(filters=7*self.channels, padding = 'same', kernel_size=(3,3), activation='relu')(encoderDown5)
        encoderDown6 = layers.Flatten()(x)
        


        yIngress = layers.Input(500, name='subDecod_input_layer');
        yFormat = layers.Dense(encoderDown6.shape[1], activation='relu')(yIngress);
        decoderUp0 = layers.Reshape((x.shape[1], x.shape[2], x.shape[3]))(yFormat);
        y = layers.Conv2DTranspose(filters=7*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp0);
        #5
        decoderUp1 = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(5,5), strides=(3,3), padding='same')(y);
        y = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp1);
        #15
        decoderUp2 = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(5,5), strides=(3,3), padding='same')(y);
        y = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp2);
        #45
        decoderUp3 = layers.Conv2DTranspose(filters=4*self.channels, kernel_size=(5,5), strides=(3,3), padding='same')(y);
        y = layers.Conv2DTranspose(filters=4*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp3);
        #135
        decoderUp4 = layers.Conv2DTranspose(filters=3*self.channels, kernel_size=(5,5), strides=(2,2), padding='same')(y);
        y = layers.Conv2DTranspose(filters=3*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp4);
        #270
        decoderUp5 = layers.Conv2DTranspose(filters=2*self.channels, kernel_size=(5,5), strides=(2,2), padding='same')(y);
        y = layers.Conv2DTranspose(filters=2*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp5);
        #540
        decoderUp6 = layers.Conv2DTranspose(filters=self.channels, kernel_size=(5,5), strides=(2,2), padding='same')(y);
        y = layers.Conv2DTranspose(filters=self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp6);
        egress = layers.Conv2D(filters=self.channels, kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid')(y);


        x_model = self.x_model = Model(ingress, encoderDown6)
        x_model._name = "subEncoder_model"
        y_model = self.y_model = Model(yIngress, egress);
        y_model._name='SubDecoder_model'

        recurrentIngress = layers.Input((self.timeframes, self.m, self.n, self.channels), name='main_input')
        recurrentEncoderWrap = layers.TimeDistributed(x_model, name='conv_timedist_layer')(recurrentIngress)
        z = bIn = layers.GRU(500, return_sequences=True)(recurrentEncoderWrap)
        z = layers.GRU(250, return_sequences=True, recurrent_dropout=0.3, dropout=0.3)(z)
        z = latentLayer = layers.GRU(10, return_sequences=True)(z)
        z = latent_in = layers.Input((self.timeframes, 10))
        z = decoderIn = layers.GRU(250, return_sequences=True, dropout= 0.3, recurrent_dropout=0.3)(z)
        z = bOut = layers.GRU(500, return_sequences=True)(z)
        recurrentDecoderWrap = layers.TimeDistributed(y_model)(z);

        encoder = Model(recurrentIngress, latentLayer)
        decoder = Model(latent_in, recurrentDecoderWrap)
        fullAutoencoder = Model(recurrentIngress, decoder(encoder(recurrentIngress)));

        return fullAutoencoder, encoder, decoder;

    def buildSegmentedAutoencoder(self):
        self.encoder = Model(self.autoencoder.get_layer('main_input'), self.autoencoder.get_layer('latent_layer'))

    # def buildNetwork2(self):
    #     ingress = layers.Input((self.timeframes, self.m, self.n, self.channels))
    #     x = layers.ConvLSTM2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same', return_sequences=True, activation='relu')(ingress)
    #     x = layers.ConvLSTM2D(filters=9, kernel_size=(3,3), strides=(2,2), padding='same', return_sequences=True, activation='relu')(x)
    #     x = layers.ConvLSTM2D(filters=12, kernel_size=(3,3), strides=(2,2), padding='same', return_sequences=True, activation='relu')(x)
    #     x = layers.ConvLSTM2D(filters=15, kernel_size=(3,3), strides=(2,2), padding='same', return_sequences=True, activation='relu')(x)
    #     x = layers.ConvLSTM2D(filters=18, kernel_size=(5,5), strides=(3,3), padding='same', return_sequences=True, activation='relu')(x)
    #     x = layers.ConvLSTM2D(filters=21, kernel_size=(5,5), strides=(3,3), padding='same', return_sequences=True, activation='relu')(x)
    #     x = layers.ConvLSTM2D(filters=24, kernel_size=(5,5), strides=(3,3), padding='same', return_sequences=True, activation='relu')(x)
    #     x = layers.Flatten()(x)

    #     z = layers.LSTM(500, return_sequences=True)(x)
    #     z = layers.LSTM(20, return_sequences=True)(z)
    #     z = layers.LSTM(500, return_sequences=True)(z)

    #     y = layers.Reshape((self.timeframes, 5, 5, 24))(z)
        


        



        
  