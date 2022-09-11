
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:39:04 2020
@author:
"""
# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from processing_functions import add_variable, get_client
from pydantic import BaseModel
import shap
#from fastapi.encoders import jsonable_encoder
#from fastapi.responses import JSONResponse
import json
from sklearn.neighbors import NearestNeighbors

# 2. Create the app object /  Initialize an instance of FastAPI
app = FastAPI()

#import model 
pickle_in = open("pipeline_bank.pkl","rb")
pipeline_process=pickle.load(pickle_in)

data_train = pd.read_csv('application_train.csv')

#Feature engineering - ADD SEVERAL VARIABLES
add_variable(data_train)

@app.get('/')
def index():
    return {'Welcome to this API for loan advisor'}

@app.get('/list')
def get_list_id():
    return data_train['SK_ID_CURR'].tolist()

#Class which describes a single id
class Item(BaseModel):
    id: int

# Information on client
@app.get('/client')
def get_client(item : Item):
    data = item.dict()
    #Select the datas with a specific client
    X = data_train[data_train['SK_ID_CURR'] == data['id']]
    return {'gender': X['CODE_GENDER'].values[0], 'credit': X['AMT_CREDIT'].values[0], 'income_tot': X['AMT_INCOME_TOTAL'].values[0], 'income_per': X['INCOME_PER_PERSON'].values[0]}

@app.post('/predict')
def predict_bank(item : Item):
    data = item.dict()
    #Select the datas with a specific client
    X = data_train[data_train['SK_ID_CURR'] == data['id']]
    notwanted_features = ['SK_ID_CURR', 'TARGET']
    selected_features = [col for col in data_train.columns if col not in notwanted_features]
    X = X[selected_features]
    prediction=pipeline_process.predict_proba(X)[:,1]
    id_score = prediction[0]
    return id_score

@app.post('/importance')
def client_featureImportance(item : Item):
    data = item.dict()
    #Select the datas with a specific client
    X = data_train[data_train['SK_ID_CURR'] == data['id']]
    X = X.drop(['TARGET','SK_ID_CURR'], axis=1)
    return X.to_json()


@app.post('/neighbors')
def client_neighbors(item : Item):
    data = item.dict()
    #Select the datas with a specific client
    X = data_train[data_train['SK_ID_CURR'] == data['id']]
    X = X.drop(['TARGET','SK_ID_CURR'], axis=1)

    #Prepare data for computation
    df_out = data_train.drop(['TARGET','SK_ID_CURR'], axis=1)
    df_prep = pipeline_process['preprocessor'].transform(df_out)
    X_prep = pipeline_process['preprocessor'].transform(X)

    #Compute 5 neighbors client
    knn = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(df_prep)
    distances, indices = knn.kneighbors(X_prep)
    indices = indices[0][1:]
    distances = distances[0][1:]
    #Compute the solvency for the neighbors client
    df_neighbors = df_out.iloc[indices, :] #Select the neighbors with indices
    pred_neighbors = pipeline_process.predict_proba(df_neighbors)[:,1]
    df = df_neighbors.copy()
    df['SOLVENCY'] = pred_neighbors
    df['CLOSENESS'] = distances
    df = df[['SOLVENCY','CLOSENESS','CODE_GENDER','AMT_CREDIT','AMT_INCOME_TOTAL','INCOME_PER_PERSON']]

    return df.to_json()

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload
