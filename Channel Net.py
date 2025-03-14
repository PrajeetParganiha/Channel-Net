# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:01:58 2025

@author: Prajeet
"""

#CHANNEL NET

import scipy.io


# Load the .mat file
mat_file = scipy.io.loadmat("C:\Main 2\Study\College\Year 2\Sem 4\AI in WLC\Channel Net\Perfect_H_40000.mat")

# View the keys (variable names) in the file
print(mat_file.keys())

# Access a specific variable
dataset = mat_file['My_perfect_H']

# Explore the variable (e.g., shape, type)
print(type(dataset))
print(dataset.shape)

'''40000 samples of image size 72x14'''

import tensorflow as tf
from tensorflow import layers,models

import numpy as np

def RMSE(target,ref):
    target=np.array(target,dtype=float)
    ref=np.array(ref,dtype=float)
    return np.sqroot(np.mean(np.square(target-ref)))

def psnr(target,ref):
    #assume RGB image
    #Compute MSE
    Error=RMSE(target,ref)
    MAX=255.
    return 20*np.log10(MAX/Error)

def interpolation(noisy,SNR,Number_of_pilot,interp):
    noisy_image=np.zeroes((40000,72,14,2))
    
    noisy_image[:,:,:,0] = np.real(noisy)
   noisy_image[:,:,:,1] = np.imag(noisy)


   if (Number_of_pilot == 48):
       idx = [14*i for i in range(1, 72,6)]+[4+14*(i) for i in range(4, 72,6)]+[7+14*(i) for i in range(1, 72,6)]+[11+14*(i) for i in range(4, 72,6)]
   elif (Number_of_pilot == 16):
       idx= [4+14*(i) for i in range(1, 72,9)]+[9+14*(i) for i in range(4, 72,9)]
   elif (Number_of_pilot == 24):
       idx = [14*i for i in range(1,72,9)]+ [6+14*i for i in range(4,72,9)]+ [11+14*i for i in range(1,72,9)]
   elif (Number_of_pilot == 8):
     idx = [4+14*(i) for  i in range(5,72,18)]+[9+14*(i) for i in range(8,72,18)]
   elif (Number_of_pilot == 36):
     idx = [14*(i) for  i in range(1,72,6)]+[6+14*(i) for i in range(4,72,6)] + [11+14*i for i in range(1,72,6)]
    
def SRCNN_model():
    model=models.Sequential()
    model.add(Conv2D(filters=32,kernel_size=(9,9),activation='relu',padding='valid',input_shape=(72,14,1)))
    model.add(layers.Conv2D(filters=64,kernel_size=(1,1),activation='relu',padding='valid',kernel_initializer='he_normal'))
    model.add(layers.Conv2D(filters=3,kernel_size=(3,3),padding='valid',kernel_initializer='he_normal')) 
    
    model.compile(optimizer='adam',loss=mean_squared_error,metrics=['mean_squared_error'])
    return model

def SRCNN_train(train_data ,train_label, val_data , val_label , channel_model , num_pilots , SNR ):
    model=SRCNN_model()
    interpolation()
    model.fit(train_data,train_label, batch_size=128 validation_data=(val_data,val_label))
    
def SRCNN_predict(input_data , channel_model , num_pilots , SNR):
    model=SRCNN_model()
    return model

def DNCNN_model():
    model=models.Sequential()
    f=
    model.add(Conv2D(filters=64,kernel_size=(3,3,c),activation='ReLU',))
    
    model.add(Conv2D(filters=64,kernel_size=(3,3,64)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('ReLU'))
    
    callbacks=[tf.callbacks.LearningRateScheduler((lambda i,lr : return (1e-1)**() if i<50 else) ),()    
               
def DNCNN_train(train_data,train_label, val_data, val_label, channel_model, num_pilots,SNR):
    model=DNCNN_model()
    model.fit(train_data,train_label,validation_data=(val_data,val_label))
    model.save_weights()