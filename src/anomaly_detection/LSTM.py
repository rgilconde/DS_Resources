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

def Scaler(df,direction):
    # Mediante esta funcion creamos y transfromamos los datos mediante un normalizador
# Standardizing the features
    scaler=StandardScaler()
    final_data = scaler.fit_transform(df)
# Download scaler
    dump(scaler, open(direction+'model_scaler.pkl', 'wb'))
    return final_data

# ETL to sequences

def create_sequences(X,time_steps):
    # X data en forma de numpy
    # time_steps, la cantidad de datos que quieres meter
    Xs=[]
    for i in range(X.shape[0]-time_steps+1):
        Xs.append(X[i:(i+time_steps)])
    return np.array(Xs)

# Create model

def model_lstm(data,time_steps,dim_out,dim_in,dp):
    # Data, es data en forma de create_sequences
    # time_steps, es la cantidad de datos de sequencia que se quiere meter
    # dim out & dim in, dimensionalidad  de la red neuronalde lstm
    # dp is drop_out
    model = Sequential()
    model.add(LSTM(dim_out, activation='relu', input_shape=(time_steps,data.shape[2]), return_sequences=True))
    model.add(Dropout(dp))
    model.add(LSTM(dim_in, activation='relu', return_sequences=True))
    model.add(Dropout(dp))
    model.add(LSTM(dim_in, activation='relu', return_sequences=True))
    model.add(Dropout(dp))
    model.add(LSTM(dim_out, activation='relu', return_sequences=True))
    model.add(Dropout(dp))
    model.add(TimeDistributed(Dense(data.shape[2])))
    model.compile(optimizer='adam', loss='mse')
    return model

# Training

def training(data,model,patience,direccion,bs,n_epochs,verbose):
    # data, model
    # patience, cantidad de epochs que no avanza para meteler checkpoints
    # bs es la cantidad de batch & n_epochs, es la cantidad maxima de epcohs que se pueden dar 
    

    
    es = EarlyStopping(monitor='loss', mode='min', verbose=verbose, patience=patience)
    mc = ModelCheckpoint(direccion+'model_lstm.h5', monitor='loss', mode='min', verbose=verbose, save_best_only=True)
    model.fit(data,data,batch_size=bs,epochs=n_epochs,verbose=verbose,callbacks=[es,mc])


# Calculate P_Value
def P_value_calculate(model_dir,data_array,porcentaje_max,direction):
	model=keras.models.load_model(model_dir)
	data_array_pred=model.predict(data_array)
	resta=np.abs(data_array_pred - data_array)
	resta_global_total=resta.mean(axis=2).mean(axis=1)
	value_limited=sorted(resta_global_total,reverse=True)[int(len(resta_global_total)*porcentaje_max)]
	np.save(direction+'p_value.npy', np.array(value_limited))
	return resta_global_total
    		    			


def Export_results_xlsx(resta_global_total,value_limited,data_select_dataframe,num_max,t_stp,direccion):

# Calculate id (del array) out/in
	resta_global_binary=[]
	for r in resta_global_total:
    		if r>value_limited:
        		resta_global_binary.append(1)
    		else:
        		resta_global_binary.append(0)
        
# Calculate los id que son out/in reales
	out_list=[]
	for i in range(0,len(resta_global_binary)):
    		if resta_global_binary[i]==1:
        		list_pre=list(range(i,i+t_stp))
        		for l in list_pre:
                		out_list.append(l)
    		else:
        		continue

	dictionary_counts=Counter(out_list)
	dictionary_keys=list(Counter(out_list).keys())
	final=[]
	for key in dictionary_keys:
    		if dictionary_counts[key]>num_max:
        		final.append(key) 

	# Merge results out/in
	Final=pd.DataFrame(final)
	Final['Out/In']=1
	Final.columns=['Index','Out/In']
	Final.set_index('Index', inplace=True)
	result = pd.merge(data_select_dataframe, Final, how="left",left_index=True, right_index=True).fillna(0)
	result.to_excel(direccion+"_resultados.xlsx")
	result=result.reset_index()
	return result

def graphs(direccion,result):
	for prop in result.columns[1:-1]:
    		fig,ax=plt.subplots(figsize=(20, 16)) 
    		plt.plot(result['index'],result[prop],marker="o",label=prop)
    		plt.plot(result[result['Out/In']==1]['index'],result[result['Out/In']==1][prop],marker="o",color="red",linestyle='')
    		plt.title(prop, fontdict=None, loc='center', pad=None,fontsize=30)
    		plt.savefig(direccion+'Grap_'+prop+'.png', bbox_inches='tight', pad_inches=0.0)


