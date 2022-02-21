#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import pickle
import numpy as np
import tensorflow as tf
import glob
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pickle import dump
import pandas as pd
from collections import Counter


# ETL Standarization

def Scaler(df,direction,name):
    # Mediante esta funcion creamos y transfromamos los datos mediante un normalizador
# Standardizing the features
    scaler=StandardScaler()
    final_data = scaler.fit_transform(np.array(df).reshape(-1, 1))
# Download scaler
    dump(scaler, open(direction+'model_scaler'+name+'.pkl', 'wb'))
    return final_data

# ETL to sequences

def create_sequences(X,time_steps,time_steps_y):
    # X data en forma de numpy
    # time_steps, la cantidad de datos que quieres meter
    Xs=[]
    ys=[]
    for i in range(X.shape[0]-time_steps-time_steps_y):
        Xs.append(X[i:(i+time_steps)])
        ys.append(X[i+time_steps:time_steps_y+i+time_steps])
    ys=np.array(ys)
    return np.array(Xs),ys.reshape(ys.shape[0],ys.shape[1])
    
# Create model








def model_lstm(X,y,dim1,dim2,dp):
	
    model = Sequential()
    model.add(LSTM(dim1, activation='relu', input_shape=(X.shape[1],X.shape[2]), return_sequences=True))
    model.add(Dropout(dp))
    model.add(LSTM(dim2, activation='relu'))
    model.add(Dropout(dp))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    return model

# Training




def training(X_train,y_train,X_test, y_test,model,patience,direccion,bs,n_epochs,verbose):
    # data, model
    # patience, cantidad de epochs que no avanza para meteler checkpoints
    # bs es la cantidad de batch & n_epochs, es la cantidad maxima de epcohs que se pueden dar 
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience)
    mc = ModelCheckpoint(direccion+'model_lstm.h5', monitor='val_loss', mode='min', verbose=verbose, save_best_only=True)
    model.fit(X_train,y_train,batch_size=bs,epochs=n_epochs,verbose=verbose,validation_data=(X_test, y_test),callbacks=[es,mc])


def graphs(X,y,scaler,lin_inf,lim_sup,model):
	plt.plot(pd.DataFrame(scaler.inverse_transform(model.predict(X[lin_inf:lim_sup]))), label = "Predict")
# plotting the line 2 points 
	plt.plot(pd.DataFrame(scaler.inverse_transform(y[lin_inf:lim_sup])), label = "Real")
# show a legend on the plot
	plt.legend()
# Display a figure.
	plt.show()


