#!/usr/bin/env python
# coding: utf-8


from tensorflow.keras.models import Sequential,Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from pickle import dump
def split_test_train(X,y,test_size):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=314)
	return X_train, X_test, y_train, y_test


# ETL Standarization

def Scaler(df,direction):
    # Mediante esta funcion creamos y transfromamos los datos mediante un normalizador
# Standardizing the features
    scaler=StandardScaler()
    final_data = scaler.fit_transform(df)
# Download scaler
    dump(scaler, open(direction+'model_scaler.pkl', 'wb'))
    return final_data

def model_autoencoder(data,dim_1,dim_2,dim_3,dp):
    model = Sequential()
    model.add(Dense(dim_1, input_dim=data.shape[1], activation='relu',name='cod1'))
    model.add(Dense(dim_2, activation='relu',name='cod2'))
    model.add(Dense(dim_3, activation='relu',name='cod3'))
    model.add(Dense(dim_3, activation='relu',name='decod3'))
    model.add(Dense(dim_2, activation='relu',name='decod2'))
    model.add(Dense(dim_3, activation='relu',name='decod1'))
    model.add(Dense(data.shape[1],name='salida'))
    model.compile(optimizer='adam', loss='mse')	
    model.summary()
    return model
    
def training(data,model,patience,direccion,bs,n_epochs,verbose):
    # data, model
    # patience, cantidad de epochs que no avanza para meteler checkpoints
    # bs es la cantidad de batch & n_epochs, es la cantidad maxima de epcohs que se pueden dar 
    

    
    es = EarlyStopping(monitor='loss', mode='min', verbose=verbose, patience=patience)
    mc = ModelCheckpoint(direccion+'model_autoencoder.h5', monitor='loss', mode='min', verbose=verbose, save_best_only=True)
    model.fit(data,data,batch_size=bs,epochs=n_epochs,verbose=verbose,callbacks=[es,mc])

