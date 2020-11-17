# csulb-datascience
#
# Author: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#
# Class version: 2.0
# Date: July 2020
#
# Include a reference to this site if you will use this code.

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from Recognition_DecoderModel_V2 import DecoderModel
from Recognition_EncoderModel_V2 import EncoderModel
from Recognition_TripletLoss_V2 import TripletSemiHardLoss
from Recognition_PrototypeLoss_V1 import PrototypeLoss


class Autoencoder:
    #constructor
    def __init__(self, unitSize, alpha=1.0, beta=1.0, lambdaVal=1.0, learningRate=0.001, kappa=1.0):
        self.unitSize = unitSize
        self.alpha = alpha
        self.beta = beta
        self.lambdaVal = lambdaVal
        self.kappa = kappa
        self.learningRate = learningRate
        self.encoderModel = EncoderModel()
        self.decoderModel = DecoderModel(self.encoderModel.embeddingSize)                
                                
    #Build the autoencoder
    def getAutoencoderCNN(self):
        #Common Input
        inputPress = Input(shape = (self.unitSize, self.encoderModel.featuresPress), name="input_press")
        inputAcc = Input(shape = (self.unitSize, self.encoderModel.featuresAcc), name="input_acc")
        inputGyro = Input(shape = (self.unitSize, self.encoderModel.featuresGyro), name="input_gyro")
        inputTriModal = [inputPress, inputAcc, inputGyro]        

        #Insert Encoder-Decoder
        embeddings = self.encoderModel.getNetworkEncoderCNN(inputTriModal) 
        decodedUnits = self.decoderModel.getNetworkDecoderCNN(embeddings, self.unitSize)
        
        #Define the models
        encoder = Model(inputs= inputTriModal, outputs=embeddings)
        autoencoder = Model(inputs= inputTriModal, outputs=[embeddings, decodedUnits])
        
        #Return the models
        return(autoencoder, encoder)
    

    def getCompiledCNN(self):
        losses = {"encoder": TripletSemiHardLoss(self.alpha, self.beta), "decoder": PrototypeLoss()}
        lossWeights = {"encoder": self.kappa, "decoder": self.lambdaVal}        
        autoencoder, encoder = self.getAutoencoderCNN()
        autoencoder.compile(optimizer=Adam(self.learningRate), loss=losses, loss_weights=lossWeights)        
        return(autoencoder, encoder)

    
    
    #****************************************************************************************************
    # OTHERS
    #****************************************************************************************************
    
    def getResults(self, history):
        #get the values
        lossTrain, lossValid, lossEnc, lossValidEnc, lossDec, lossValidDec = -1, -1, -1, -1, -1, -1
        if "loss" in history.history.keys(): lossTrain = history.history["loss"][-1]
        if "val_loss" in history.history.keys(): lossValid = history.history["val_loss"][-1]
        if "encoder_loss" in history.history.keys(): lossEnc = history.history["encoder_loss"][-1]
        if "val_encoder_loss" in history.history.keys(): lossValidEnc = history.history["val_encoder_loss"][-1]            
        if "decoder_loss" in history.history.keys(): lossDec = history.history["decoder_loss"][-1]
        if "val_decoder_loss" in history.history.keys(): lossValidDec = history.history["val_decoder_loss"][-1]                        
        return(lossTrain, lossValid, lossEnc, lossValidEnc, lossDec, lossValidDec)    
