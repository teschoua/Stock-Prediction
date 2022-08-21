import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import statistics
import copy

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM

# @st.cache
def load_data():
    
    tickerSymbols = "TSLA"

    data  = yf.download(tickers = tickerSymbols,
                        start = "2000-01-01",
                        end = "2022-08-17",
                        interval = "1d",
                        group_by = 'ticker',
                        # Mass Downloading
                        threads = True,
                        proxy = None
                        )
    return data

# Load Model LSTM
def load_model(name_model):
    return tf.keras.models.load_model(name_model)    

# Split Df on stock names
def split_df(df):

    # Split on stock names (Faire une fonction qui gere les names dynamiquement)
    df_TSLA = df.iloc[:, 0:6]
    df_AMZN = df.iloc[:, 6:12]
    df_GGL = df.iloc[:, 12:18]
    df_MSFT = df.iloc[:, 18:24]
    df_AAPL = df.iloc[:, 24:30]

    return df_TSLA, df_AMZN, df_GGL, df_MSFT, df_AAPL

# Scale Data (Adjusted Close column)
def scale_data(df):
    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Adjusted Close index
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Adjusted Close index
    df_log = pd.DataFrame(data=df_log, index=df.index)
    df_log.rename(columns={0:'Adj Close'}, inplace=True)
    return df_log

# Preprocessing
""" 
    Prend n prix d'une série temporelle (n_min = 30 jours)
    Renvoie une liste de liste de 30 jours de prix (dans l'ordre chronologique)
    Exemple : input = [1,2,3,...,37] -> 
              output = [ [1,2,...,30],
                         [2,3,...,31],
                         [...],
                         [7,8,...,37] ]
"""
def preprocess_multistep_lstm(sequence, n_steps_in=30, n_steps_out=7, n_features=1):
    X = list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out   
        # Check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # Gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        X.append(seq_x)

    X = np.array(X)

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    return X

# Split dataset
def split_train_test(X, y, split_percentage=0.8):

    training_data_len = math.ceil(len(X) * split_percentage)

    X_train, y_train = X[:training_data_len], y[:training_data_len]
    X_test, y_test = X[training_data_len:], y[training_data_len:]

    return X_train, y_train, X_test, y_test

# model = load_model('Models/LSTM_epochs_15_batch_32')

def show_prediction_page():

    # Load the dataset
    df = load_data()

    # Scale Data
    df_scaled = scale_data(df)

    # Preprocess input data
    data_preprocessed = preprocess_multistep_lstm(df_scaled)

    print(type(data_preprocessed))
    st.write(data_preprocessed)

































