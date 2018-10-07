# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:14:09 2018

@author: zxs123
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

## generate visional data
import numpy as np
data = np.random.random((1000,100))
labels = np.random.randint(10,size=(1000,1))

one_hot_labels = keras.utils.to_categorical(labels,num_classes=10)

#train
model.fit(data,one_hot_labels,epochs=10,batch_size=32)