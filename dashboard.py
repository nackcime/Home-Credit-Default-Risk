
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:39:04 2020
@author:
"""
# 1. Library imports
import requests
import json
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import pickle

#import model 
pickle_in = open("pipeline_bank.pkl","rb")
pipeline_process=pickle.load(pickle_in)

# 2. Create the app object /  Initialize an instance of FastAPI
#chart = functools.partial(st.plotly_chart, use_container_width=True)

data_train = pd.read_csv('application_train.csv')
headers = {'Content-Type': 'application/json'}
#url_id = "http://127.0.0.1:8000/predict"
#url_list = "http://127.0.0.1:8000/list"
url_id = "https://loan-advisor-api.herokuapp.com/predict"
url_list = "https://loan-advisor-api.herokuapp.com/list"
url_client = 'https://loan-advisor-api.herokuapp.com/client'
url_importance = 'https://loan-advisor-api.herokuapp.com/importance'
url_neighbors = 'https://loan-advisor-api.herokuapp.com/neighbors'

#"""API - load list of clients""" 
id_list_response = requests.get(url_list, headers=headers)
id_list = json.loads(id_list_response.content)  #parse bytes data with json module

def main():
    st.title("Bank Helper")
    

    """Select a client in the list"""
    id_client = st.selectbox('Choose Id client', id_list, help = 'Filter report to show only one id client')
    id_client_json = {"id": id_client}
    
    #Show information about prediction
    with st.expander("Solvency", expanded=False):
        pred = requests.post(url_id, headers=headers, json=id_client_json)
        result = float(pred.content)
        st.success('Your solvency is at {:.0%}'.format(round(result, 4)))
        id_score = [1-result,result]
        names=['Not Creditworthy', 'Creditworthy']
        st.subheader("Creditworthy Rate")
        fig = px.pie(values = id_score,names=names)
        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = result,
            mode = "gauge+number+delta",
            title = {'text': "Solvency"},
            #delta = {'reference': 380},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [
                         {'range': [0, 20], 'color': "red"},
                         {'range': [20, 50], 'color': "yellow"}],
                         {'range': [50, 100], 'color': "green"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 10}}))
        
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
        
    #Show client information
    with st.expander("Client Information", expanded=False):
        client_info_link = requests.get(url_client, headers=headers, json=id_client_json)
        client_info = json.loads(client_info_link.content)
        df_client_info = pd.DataFrame([client_info])
        st.write(df_client_info)
        chck_neighbors = st.checkbox("Show similar cases ?")
        
    #Show information about close clients
    if chck_neighbors:
        client_neighbors_link = requests.post(url_neighbors, headers=headers, json=id_client_json)
        client_neighbors = json.loads(client_neighbors_link.content)
        df_client_neighbors = pd.read_json(client_neighbors)
        st.markdown("<u>List of the 5 closest client :</u>", unsafe_allow_html=True)
        st.write(df_client_neighbors)
    else:
        st.markdown("<i>Masqued informations</i>", unsafe_allow_html=True)
        
    #Show information local feature importance - Shap
    with st.expander("More Information - Feature importance", expanded=False):
        importance_info_link = requests.post(url_importance, headers=headers, json=id_client_json)
        importance_info = json.loads(importance_info_link.content)
        df_importance_info = pd.read_json(importance_info)
        columnsEncoded = pipeline_process['preprocessor'].transformers_[0][1][1].get_feature_names_out()
        columnsAll = columnsEncoded.tolist() + df_importance_info.columns.tolist()

        observations_preprocessed = pipeline_process['preprocessor'].transform(df_importance_info)
        explainer = shap.TreeExplainer(pipeline_process['classifier'])
        shap_values = explainer.shap_values(observations_preprocessed)
        st_shap(shap.summary_plot(shap_values[0][:], observations_preprocessed,feature_names=columnsAll, max_display=10))


        
if __name__=='__main__':
    main()
