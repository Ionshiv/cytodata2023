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
        # self.autoencoder = self.buildFullAutoencoder();
        self.autoencoder = self.buildTest();
        self.latentOutputLayer = self.autoencoder.get_layer(name='latent_layer');
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
    
    def getLatentSpaceLayer(self):
        return self.latentOutputLayer;

    def compileAutoencoder(self):
        self.autoencoder.compile(optimizer='adam', loss='mse');

    def getSummaryAutoencoder(self):
        self.x_model.summary();
        self.autoencoder.summary();
        self.y_model.summary();
    
    def fitAutoencoder(self):
        aug_factor = 1
        if self.input_aug:
            aug_factor = 4;
        history = self.autoencoder.fit(x=self.lcpGen.trainGen(), epochs=3, batch_size=aug_factor*self.batch_size, steps_per_epoch=aug_factor*len(self.lcpGen.trainpath)//self.batch_size, verbose=1, validation_data=self.lcpGen.validGen(), validation_steps=aug_factor*len(self.lcpGen.validpath)//self.batch_size)
        return history;

    def buildFullAutoencoder(self):
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
        x_model = self.x_model = Model(ingress, encoderDown6)
        x_model._name = "subEncoder_model"
        rEnIn = layers.Input((self.timeframes, self.m, self.n, self.channels), name='conv_time_input_layer')
        rEncoderSegment = layers.TimeDistributed(x_model, name='conv_timedist_layer')(rEnIn)


        bottleneck = bIn = layers.GRU(500, return_sequences=True)(rEncoderSegment)
        bottleneck = layers.GRU(250, return_sequences=True)(bottleneck)
        bottleneck = latentLayer = layers.GRU(100, return_sequences=True, name='latent_layer')(bottleneck)
        bottleneck = layers.GRU(250, return_sequences=True)(bottleneck)
        bottleneck = bOut = layers.GRU(500, return_sequences=True)(bottleneck)


        dIngress = layers.Input(bOut.shape[2], name='subDecod_input_layer');
        dFormat = layers.Dense(encoderDown6.shape[1], activation='relu')(dIngress);
        decoderUp0 = layers.Reshape((encoder.shape[1], encoder.shape[2], encoder.shape[3]))(dFormat);
        decoder = layers.Conv2DTranspose(filters=7*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp0);
        decoder = layers.Conv2DTranspose(filters=7*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=7*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);

        decoderUp1 = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), strides=(3,3), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp1);
        decoder = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=6*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);

        decoderUp2 = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), strides=(3,3), padding='same')(decoder);
        decoder = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoderUp2);
        decoder = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);
        decoder = layers.Conv2DTranspose(filters=5*self.channels, kernel_size=(3,3), padding='same', activation='relu')(decoder);

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
        

        y_model = self.y_model = Model(dIngress, egress);
        y_model._name='SubDecoder_model'
        rDecoderSegment = layers.TimeDistributed(y_model)(bOut);

        fullAutoencoder = Model(rEnIn, rDecoderSegment);
        return fullAutoencoder;

    def buildSegmentedAutoencoder(self):
        print('TBD')

    def buildTest(self):
        print('Dormant')
        input1 = layers.Input((1080, 1080, 6))
        x = layers.Conv2D(6, (3,3), padding='same', activation='relu')(input1)
        x = layers.MaxPooling2D(pool_size=(4,4), strides=(4, 4))(x)
        x = layers.Conv2D(6, (3,3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2))(x)
        x = layers.Conv2D(6, (3,3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(9,9), strides=(9, 9))(x)
        x_flat = layers.Flatten()(x)

        x_model = self.x_model = Model(input1, x_flat)

        # x_model.summary()

        input3 = layers.Input((500))
        y = layers.Dense(1350, activation='relu')(input3)
        y = layers.Reshape((15, 15, 6))(y)
        y = layers.Conv2DTranspose(filters=6, kernel_size=(9,9), strides=(9,9), padding='same', activation='relu')(y)
        y = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(y)
        y = layers.Conv2DTranspose(filters=6, kernel_size=(2,2), strides=(2,2), padding='same', activation='relu')(y)
        y = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(y)
        y = layers.Conv2DTranspose(filters=6, kernel_size=(4,4), strides=(4,4), padding='same', activation='relu')(y)
        y_out = layers.Conv2DTranspose(filters=6, kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid')(y)

        y_model = self.y_model =  Model(input3, y_out)

        # y_model.summary()

        input2 = layers.Input((25, 1080, 1080, 6))
        r = layers.TimeDistributed(x_model)(input2)
        r = layers.GRU(500, return_sequences=True, name='latent_layer')(r)
        r = layers.Dropout(0.5)(r)
        r_out = layers.TimeDistributed(y_model)(r)

        r_model =  Model(input2, r_out)

        # r_model.summary()

        return r_model




        



        
  