# csulb-datascience
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
#
# Class version: 2.0
# Date: July 2020
#
# Include a reference to this site if you will use this code.

import keras as keras
import numpy as np

from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import concatenate
from keras.layers import Lambda
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from Recognition_PrototypeLoss_V1 import PrototypeLoss

class DecoderModel:
    def __init__(self, embeddingSize):
        self.embeddingSize = embeddingSize
        self.numModal = 3
        self.featuresPress = 16
        self.featuresAcc = 6
        self.featuresGyro = 6
    
    def Conv1DTranspose(self, input_tensor, filters, kernel_size, strides=1, padding='same', activation="relu"):
        x = Lambda(lambda x: keras.backend.expand_dims(x, axis=2))(input_tensor)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding, activation=activation)(x)
        x = Lambda(lambda x: keras.backend.squeeze(x, axis=2))(x)
        return x

    #Decoder for a sensor branch
    def getBranchCNN(self, inputs, unitSize, features):
        #Unflatten the input to a 2D shape
        x = Reshape((unitSize, inputs.shape[1].value//unitSize))(inputs)
        x = BatchNormalization()(x)
    
        # Reverse the convolution
        x = self.Conv1DTranspose(x, filters=128, kernel_size=20, padding='same', activation="relu")
        x = BatchNormalization()(x)
        
        # Reverse the convolution
        x = self.Conv1DTranspose(x, filters=64, kernel_size=20, padding='same', activation="relu")
        x = BatchNormalization()(x)
        
        # Reverse the convolution
        x = self.Conv1DTranspose(x, filters=32, kernel_size=20, padding='same', activation="relu")
        x = BatchNormalization()(x)
    
        #Back to the original shape
        x = Conv1D(features, kernel_size=20, activation='tanh', padding='same')(x)
        return (x)
    
    
    #Decoder of the multimodal network
    def getNetworkDecoderCNN(self, inputs, unitSize):
        #base layer -> reverse the encoder
        x = Dense(256, activation="relu")(inputs)
        x = BatchNormalization()(x)
            
        #reverse the size of the concatenation
        length = unitSize * 128
        x = Dense(length * self.numModal, activation="relu")(x)
        x = BatchNormalization()(x)
    
        #Split the tensor for each modal
        press = Lambda(lambda x: x[:, 0: length])(x)
        acc = Lambda(lambda x: x[:, length: 2*length])(x)
        gyro = Lambda(lambda x: x[:, 2*length: 3*length])(x)
        
        #Obtain the unit
        press = self.getBranchCNN(press, unitSize, self.featuresPress)
        acc = self.getBranchCNN(acc, unitSize, self.featuresAcc)
        gyro = self.getBranchCNN(gyro, unitSize, self.featuresGyro)        
        unit = concatenate([press, acc, gyro], axis=2, name="unit_step")
        unit = Lambda(lambda x: keras.backend.l2_normalize(x, axis=1), name="decoder")(unit)                
        return(unit)       
    
    #Returns the model of the decoder
    def getDecoderCNN(self, unitSize):            
        #base layer -> reverse the encoder
        inputs = Input(shape= self.embeddingSize, name="input_decoder")
        unit = self.getNetworkDecoderCNN(inputs, unitSize)
        model = Model(inputs=inputs, outputs=unit)
        return(model)
          
    #Returns the compiled Decoder model
    def getCompiledCNN(self, unitSize):
        model = self.getDecoderCNN(unitSize)
        model.compile(loss=PrototypeLoss(), optimizer = Adam(), metrics=['accuracy'])
        return(model)     

    #****************************************************************************************************
    # OTHERS
    #****************************************************************************************************
    
    def getResults(self, history):
        #get the values
        lossTrain, lossValid = -1, -1
        if "loss" in history.history.keys(): lossTrain = history.history["loss"][-1]
        if "val_loss" in history.history.keys(): lossValid = history.history["val_loss"][-1]        
        return(lossTrain, lossValid)    
            