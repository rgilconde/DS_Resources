#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pickle import dump
import matplotlib.pyplot as plt
import pandas as pd
from shaphypetune import BoostSearch, BoostBoruta, BoostRFE, BoostRFA
from xgboost import *


# ETL to sequences

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



# Create model

def model_boosting(X_train_scaler,y_train,X_test_scaler,y_test,param_grid):
	regr_xgb = XGBRegressor(n_estimators=150000, random_state=0, verbosity=0, n_jobs=-1)
	model = BoostSearch(regr_xgb, param_grid=param_grid)
	model.fit(X_train_scaler, y_train, eval_set=[(X_test_scaler, y_test)], early_stopping_rounds=10, verbose=1)
	return model.best_params_

# Training

def training_results(X_train_scaler,X_test_scaler,y_train,y_test,direccion_results,direction_models,eta,num_leaves,max_depth,colsample_by,sub_sample,min_child_weight):
	xgb_final=XGBRegressor(n_estimators=15000, random_state=0, verbosity=0, n_jobs=-1,eta=eta,num_leaves=num_leaves,max_depth=max_depth,colsample_by=colsample_by,sub_sample=sub_sample,min_child_weight=min_child_weight)
	xgb_final.fit(X_train_scaler, y_train, eval_set=[(X_test_scaler, y_test)], early_stopping_rounds=10, verbose=1)
	results=pd.DataFrame(xgb_final.predict(X_train_scaler))
	real=y_train.reset_index(drop=True)
	final=pd.concat([real,results],axis=1)
	final.columns=['Volume','Volume_Pre']
	final.to_excel(direccion_results+"results_boosting.xlsx")
	dump(xgb_final, open(direction_models+'model_xgb.pkl', 'wb'))
	return final
	

def graphs(direccion,result,ini,final,name):
		final_dib=result.iloc[ini:final]
		fig,ax=plt.subplots(figsize=(20, 16)) 
		plt.plot(final_dib[name],color="black",marker="o",label=name)
		plt.plot(final_dib[name+'_Pre'],marker="o",color="blue",label=name+'_Pre')
		ax.legend()
		plt.title(name, fontdict=None, loc='center', pad=None,fontsize=30)


