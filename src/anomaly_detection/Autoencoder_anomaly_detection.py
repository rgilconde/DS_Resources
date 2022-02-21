#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint
import keras
import pickle
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pickle import dump
import pandas as pd
from collections import Counter


# ETL Standarization

def Scaler(df,direction):
    # Mediante esta funcion creamos y transfromamos los datos mediante un normalizador
# Standardizing the features
    scaler=StandardScaler()
    final_data = scaler.fit_transform(df)
# Download scaler
    dump(scaler, open(direction+'model_scaler.pkl', 'wb'))
    return final_data

# ETL to sequences


# Create model





def model_autoencoder(data,dim_1,dim_2,dim_3,dp):
    model = Sequential()
    model.add(Dense(dim_1, input_dim=data.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(dim_2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(dim_3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(dim_3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(dim_2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(dim_1, activation='relu'))
    model.add(Dense(data.shape[1]))
    model.compile(optimizer='adam', loss='mse')	
    return model
	
    # Data, es data en forma de create_sequences
    # time_steps, es la cantidad de datos de sequencia que se quiere meter
    # dim out & dim in, dimensionalidad  de la red neuronalde lstm
    # dp is drop_out


# Training

def training(data,val,model,patience,direccion,bs,n_epochs,verbose):
    # data, model
    # patience, cantidad de epochs que no avanza para meteler checkpoints
    # bs es la cantidad de batch & n_epochs, es la cantidad maxima de epcohs que se pueden dar 
    

    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience)
    mc = ModelCheckpoint(direccion+'model_autoencoder.h5', monitor='val_loss', mode='min', verbose=verbose, save_best_only=True)
    model.fit(data,data,batch_size=bs,epochs=n_epochs,verbose=verbose,callbacks=[es,mc],validation_data=(val,val))


# Calculate P_Value
def P_value_calculate(model_dir,data_array,porcentaje_max,direction):
	model=keras.models.load_model(model_dir)
	data_array_pred=model.predict(data_array)
	resta=np.abs(data_array_pred - data_array)
	resta_global_total=resta.mean(axis=1)
	value_limited=sorted(resta_global_total,reverse=True)[int(len(resta_global_total)*porcentaje_max)]
	np.save(direction+'p_value.npy', np.array(value_limited))
	return resta_global_total
    		    			


def Export_results_xlsx(resta_global_total,value_limited,data_select_dataframe,direccion):

# Calculate id (del array) out/in
	resta_global_binary=[]
	for r in resta_global_total:
    		if r>value_limited:
        		resta_global_binary.append(1)
    		else:
        		resta_global_binary.append(0)
        
	Final=pd.DataFrame(resta_global_binary)
	Final.columns=['Out/In']
	result = pd.merge(data_select_dataframe, Final, how="left",left_index=True, right_index=True).fillna(0)
	
	result.to_excel(direccion+"_resultados.xlsx")
	return result

def graphs(direccion,result):
	for prop in result.columns[1:-1]:
    		fig,ax=plt.subplots(figsize=(20, 16)) 
    		plt.plot(result['index'],result[prop],marker="o",label=prop)
    		plt.plot(result[result['Out/In']==1]['index'],result[result['Out/In']==1][prop],marker="o",color="red",linestyle='')
    		plt.title(prop, fontdict=None, loc='center', pad=None,fontsize=30)
    		plt.savefig(direccion+'Grap_'+prop+'.png', bbox_inches='tight', pad_inches=0.0)


