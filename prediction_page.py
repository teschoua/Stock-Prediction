import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import statistics
import mplfinance as mpf
import plotly.graph_objects as go
from datetime import datetime

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
    # data.to_csv('Dataset/TSLA_2010_1d.csv')
    return data

# Load Model LSTM
def load_model(name_model):
    return tf.keras.models.load_model(name_model)    

# Scale Data (Adjusted Close column)
def scale_data(df):
    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Adjusted Close index
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Adjusted Close index
    df_log = pd.DataFrame(data=df_log, index=df.index)
    df_log.rename(columns={0:'Adj Close'}, inplace=True)
    return df_log, minmax

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

def simulator(predictions,
              real_values_stock,
              n_steps_out = 7,
              initial_inventory_money = 10000,
              max_buy = 1, # Nb max of stock to buy 
              max_sell = 1 # Nb max of stock to sell
              ):

    # Convert df real_values_stock into an array 
    array_real_values_stock = np.array(real_values_stock['Adj Close'])[30:]

    # Keeps track of all the trades (buy/sell)
    df_historic_trades = pd.DataFrame(columns={'inventory_money',
                                            'buy_day',
                                            'sell_day', 
                                            'buy_price',
                                            'sell_price'
                                               }
                                    ) 

    # Keeps track of what happened for each trade
    logs = [] 

    # Balance inventory money equals at the begining to the initial_inventory_money
    inventory_money = initial_inventory_money

    def buy(profit, inventory_money, df_historic_trades): # profit : [value_profit_predicted, buy_day, sell_day]
        
        predicted_price = profit[0]
        buy_day = profit[1]
        sell_day = profit[2]
        # Retrieve the timestamp of the buy/sell day, for the df
        buy_day_format_datetime = real_values_stock.iloc[buy_day + 30].name
        sell_day_format_datetime = real_values_stock.iloc[sell_day + 30].name
        # Retrieve the actual price for buy/sell
        buy_price = array_real_values_stock[buy_day]
        sell_price = array_real_values_stock[sell_day]

        # Not enough money to buy
        if inventory_money < buy_price * max_buy :
            new_log += f"Day {buy_day} : Total balance : {inventory_money}, Not enough money to buy {max_buy} stock(s) at {buy_price}$"
            new_log += "\n"
            logs.append(new_log)
            # print(f"Day {buy_day} : Total balance : {inventory_money}, Not enough money to buy {max_buy} stock(s) at {buy_price}$ \n")
        
        else :
            inventory_money += (sell_price * max_sell) -(buy_price * max_buy)

            new_row = pd.DataFrame({'inventory_money' : inventory_money, 
                                    'buy_day' : pd.to_datetime(buy_day_format_datetime).date(), 
                                    'buy_price' : buy_price, 
                                    'sell_day' : pd.to_datetime(sell_day_format_datetime).date(), 
                                    'sell_price' : sell_price
                                    },
                                    index=[0]
                                  )
            # df_historic_trades = pd.concat([new_row,df_historic_trades.loc[:]]).reset_index(drop=True)
            df_historic_trades = pd.concat([df_historic_trades, new_row], axis=0, ignore_index=True)


        return inventory_money, df_historic_trades

    i = 0
    while i + n_steps_out < len(predictions) :
        # Update logs of trades
        new_log = f"Day {i} - Balance : {inventory_money}, Profits : {((inventory_money/initial_inventory_money) - 1) * 100:.2f}%" + "\n"
        logs.append(new_log)

        sell_day = None
        profit = [predictions[i+1]-predictions[i], i, i+1] # profit = [value_profit_predicted, buy_day, sell_day]

        # Find the max difference value in the interval [i : i+7]
        for j in range(i, i+n_steps_out-1):
            for h in range(j+1, i+n_steps_out):
                profit = [predictions[h]-predictions[j], j, h] if predictions[h]-predictions[j] > profit[0] else profit
                sell_day = h
                
        inventory_money, df_historic_trades = buy(profit, inventory_money, df_historic_trades)

        i = sell_day

    return df_historic_trades, logs

# Plot trades on real time serie
def plot_trades(df, df_historic_trades, mean_predictions):

    fig = go.Figure()
    
    # Chart (OHLC) of the real prices
    fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='Stock'
                    ),
                )

    # Buy Trades
    fig.add_trace(go.Scatter(x=df_historic_trades['buy_day'], y=df_historic_trades['buy_price'],
                             mode='markers+text',
                             name='Buy',
                             line=dict(color='rgb(8,153,129)')
                            )
                )

    # Sell Trades
    fig.add_trace(go.Scatter(x=df_historic_trades['sell_day'], y=df_historic_trades['sell_price'],
                             mode='markers+text',
                             # text='Sell',
                             # textposition="bottom center",
                             name='Sell',
                             line=dict(color='rgb(242,54,9)'),
                            #  hover_data=df_historic_trades['sell_price']
                             
                             )
                )

    # Real close prices
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['Close'],
                             name='Real Close Price',
                             line=dict(color='rgb(41, 98, 255)')
                             )
                 )

    # Predicted close prices
    df_predictions_datetime = pd.DataFrame({'Predicted Close Price' : mean_predictions}, 
                                             index = df[30:].index 
                                          )
    fig.add_trace(go.Scatter(x=df_predictions_datetime.index, 
                             y=df_predictions_datetime['Predicted Close Price'],
                             name='Predicted Close Price'
                             
                             )
                 )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        width=1200,
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font = dict(size = 18, color = "black")
        ),
        plot_bgcolor='rgb(255,255,255)',
        dragmode='pan',
        hovermode = "x unified",
        # hoverinfo="name+z"
        
    )
    fig.update_xaxes(fixedrange = False, showline=True, linewidth=2.5, linecolor='rgb(242,243,243)', gridcolor='rgb(242,243,243)')
    fig.update_yaxes(fixedrange = False, type="log", showline=True, linewidth=2.5, linecolor='rgb(242,243,243)', gridcolor='rgb(242,243,243)')
    
    return fig

# Calculate metrics for the Dashboard
def calculate_metrics_dashboard(df, initial_inventory_money=10000):
    balance = round(df.iloc[-1:]['inventory_money'].values[0], 2)
    nb_trades = round(len(df), 2)
    profits = round(balance - initial_inventory_money, 2)
    profits_percentage = round(((balance/initial_inventory_money) - 1) * 100, 2)
    period_trading = [df.iloc[0]['buy_day'], df.iloc[-1]['sell_day']]

    return balance, profits, profits_percentage, nb_trades, period_trading

def show_prediction_page():

#----------------------------------------------#
# Title (Currency Selection)
    with open('style.css') as f :
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5, gap="small")
    page = col3.selectbox('Choose stock', ('TSLA/USD', 'BTC/USD'))
     


#----------------------------------------------#
# Load and Process the data

    # Load the dataset
    df = load_data()

    # Scale Data
    df_scaled, scale_minmax = scale_data(df)

    # Preprocess input data
    data_preprocessed = preprocess_multistep_lstm(df_scaled)
    
    # Load Model LSTM
    model = load_model('Models/LSTM_epochs_15_batch_32')

    # Predict and Scale
    predictions = model.predict(data_preprocessed)
    # Inverse Scale
    predictions_unscaled = scale_minmax.inverse_transform(predictions)

    # Mean on predictions on the same day
    mean_predictions = [statistics.mean(predictions_unscaled[::-1,:].diagonal(i)) 
                        for i in range(-predictions_unscaled.shape[0]+1,predictions_unscaled.shape[1])
                       ]
    mean_predictions = np.array(mean_predictions)
    # Simulator
    df_historic_trades, logs_trades = simulator(mean_predictions, df)

#----------------------------------------------#
# Display Metrics : Currencies, Equity, Balance, Number of Trades

    col1, col2, col3, col4 = st.columns(4, gap="small")

    balance, profits, profits_percentage, nb_trades, trading_period = calculate_metrics_dashboard(df_historic_trades)

    col1.metric("Balance", str(balance) + "$")
    col2.metric("Profits", str(profits) + "$", str(profits_percentage) + "%")
    col3.metric("Trades", nb_trades)
    col4.metric("Trading Period (Y/M/D)", 
                str(trading_period[0].year) + "." + 
                str(trading_period[0].month) + "." + 
                str(trading_period[0].day) + " → " + 
                str(trading_period[1].year) + "." + 
                str(trading_period[1].month) + "." + 
                str(trading_period[1].day),
                str((trading_period[1]-trading_period[0]).days) + ' days'
                )

#----------------------------------------------#
# Display candlestick chart

    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1_3, col2_3 = st.columns([3,2], gap="small")

    fig = plot_trades(df, df_historic_trades, mean_predictions)
    col1_3.plotly_chart(fig, use_container_width=False, config=dict({'scrollZoom': True}))
    
    col2_3.dataframe(df_historic_trades[['buy_day', 'buy_price', 'sell_day', 'sell_price', 'inventory_money' ]],
                    width=800,
                    height=350
                    )
   






























