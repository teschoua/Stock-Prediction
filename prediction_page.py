from select import select
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
from datetime import datetime, timedelta

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM

class DateError(Exception):
    pass

stocks = {
            "TSLA": "TSLA/USD", 
            "AAPL": "AAPL/USD",
            "AMZN": "AMZN/USD", 
            "GOOGL": "GOOGL/USD",
            "META": "META/USD", 
            "NVDA": "NVDA/USD",

         }

dict_models = {
                "TSLA": "TSLA_BiLSTM_epochs_15_batchsize_32",
                "AAPL": "AAPL_BiLSTM_epochs_15_batchsize_32",
                "AMZN": "AMZN_BiLSTM_epochs_15_batchsize_32",
                "GOOGL": "GOOGL_BiLSTM_epochs_15_batchsize_32",
                "META": "META_BiLSTM_epochs_15_batchsize_32",
                "NVDA": "NVDA_BiLSTM_epochs_15_batchsize_32",
              }

# @st.cache
def load_data(tickerSymbols, start_date, end_date=str(datetime.now().date()), ):
    
    data = yf.download(tickers=tickerSymbols,
                       start=start_date,
                       end=end_date,
                       interval="1d",
                       group_by='ticker',
                       # Mass Downloading
                       threads=True,
                       proxy=None
                       )
    # data.to_csv('Dataset/TSLA_2010_1d.csv')
    return data

# Load Model LSTM
def load_model(name_model):
    return tf.keras.models.load_model(name_model)    

# Scale Data (Adjusted Close column)
def scale_data(df):
    # Adjusted Close index
    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32'))
    df_log = pd.DataFrame(data=df_log, index=df.index)
    df_log.rename(columns={0: 'Adj Close'}, inplace=True)
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
              initial_inventory_money,
              n_steps_out=7,
              max_buy=10, # Nb max of stock to buy
              max_sell=10 # Nb max of stock to sell
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
            new_log = f"Day {buy_day} : Total balance : {inventory_money}, Not enough money to buy {max_buy} stock(s) at {buy_price}$"
            new_log += "\n"
            logs.append(new_log)

        else:
            inventory_money += (sell_price * max_sell) -(buy_price * max_buy)

            new_row = pd.DataFrame({'inventory_money': inventory_money,
                                    'buy_day': pd.to_datetime(buy_day_format_datetime).date(),
                                    'buy_price': buy_price,
                                    'sell_day': pd.to_datetime(sell_day_format_datetime).date(),
                                    'sell_price': sell_price
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

        # Initialization
        sell_day = None
        profit = [predictions[i+1]-predictions[i], i, i+1] # profit = [value_profit_predicted, buy_day, sell_day]

        # Find the max difference value in the interval [i : i+7]
        # Loop until i+n_steps_out-1 : -1 because we don't need for the pointer j to look for the last value
        for j in range(i, i+n_steps_out-1):
            for h in range(j+1, i+n_steps_out):
                profit = [predictions[h]-predictions[j], j, h] if predictions[h]-predictions[j] > profit[0] else profit
                sell_day = h
                
        
        # No trade if the value of profit predicted is negative
        if profit[0] <= 0 : 
            i += 1
        else :
            inventory_money, df_historic_trades = buy(profit, inventory_money, df_historic_trades)

            i = sell_day + 1

    return df_historic_trades, logs

def calculate_tendency(df):
    
    df['Tendency'] = round(((df['sell_price'] - df['buy_price']) / (df['buy_price'])) * 100, 2) 
    return df

# Calculate metrics for the Dashboard
def calculate_metrics_dashboard(df, initial_inventory_money):

    balance = round(df.iloc[-1:]['inventory_money'].values[0], 2)
    nb_trades = round(len(df), 2)
    profits = round(balance - initial_inventory_money, 2)
    profits_percentage = round(((balance/initial_inventory_money) - 1) * 100, 2)
    period_trading = [df.iloc[0]['buy_day'], df.iloc[-1]['sell_day']]

    return balance, profits, profits_percentage, nb_trades, period_trading

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
    fig.add_trace(go.Scatter(
                            #  x=df.iloc[30-5 : 30-5+len(mean_predictions)].index, 
                            #  y=df['Close'][30-5:30-5+len(mean_predictions)],
                             x = df.index,
                             y=df['Close'],
                             name='Real Close Price',
                             line=dict(color='rgb(41, 98, 255)')
                             )
                 )

    # Predicted close prices
    df_predictions_datetime = pd.DataFrame({'Predicted Close Price' : mean_predictions}, 
                                            index = df.index[30:]
                                            #  index = df[30-5:30-5+len(mean_predictions)].index 
                                          )

    fig.add_trace(go.Scatter(x=df_predictions_datetime.index, 
                             y=df_predictions_datetime['Predicted Close Price'],
                             name='Predicted Close Price'
                             
                             )
                )

    fig.update_layout(
        # title = {'text': "Simulator Results", 
        #          'font_family': "Source Sans Pro", 
        #          'font': dict(size=30),
        #          'xanchor' : 'left'
        #         },
        xaxis_rangeslider_visible=False,
        # width=1200,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font = dict(size = 18, color = "black"),
        ),
        plot_bgcolor='rgb(255,255,255)',
        dragmode='pan',
        hovermode = "x unified",
        
        # hoverinfo="name+z"
        
    )

    # Axes
    fig.update_xaxes(fixedrange = False, showline=True, linewidth=2.5, linecolor='rgb(242,243,243)', gridcolor='rgb(242,243,243)', showspikes=True)
    fig.update_yaxes(fixedrange = False, type="log", showline=True, linewidth=2.5, linecolor='rgb(242,243,243)', gridcolor='rgb(242,243,243)', showspikes=True)
    
    return fig

def plot_table_trades(df, selector_stock):

    df['buy_price'] = round(df['buy_price'], 2)
    df['sell_price'] = round(df['sell_price'], 2)
    df['inventory_money'] = round(df['inventory_money'], 2)

    # Colorate tendency (positive : green, negative : red)
    df['color_row'] = df['Tendency'].apply(lambda x: '#1cac34' if x >= 0 else '#e14141')

    fill_color = []
    n = len(df)
    for col in df.columns:
        if col!='Tendency':
            fill_color.append(['rgb(242,243,243)']*n)
            # fill_color.append(['black']*n)
        else:
            fill_color.append(df["color_row"].to_list())

    # Add up/down arrow on Tendency column
    df['Tendency'] = df['Tendency'].apply(lambda x: u'\u2191' + ' ' + str(x) if x >= 0 else u'\u2193' + ' ' + str(x))
    
    fig = go.Figure(data=[go.Table(
                        # columnwidth = [133],
                        header=dict(values=["Buy Day", "Buy Price ($)", "Sell Day", "Sell Price ($)", "Balance ($)", "Tendency (%)"],
                                    fill_color='rgb(65, 105, 225)',
                                    line_color='white',
                                    align='center',
                                    font=dict(color='white', size=17),
                                   ),
                        cells=dict(values=[df.buy_day, df.buy_price, df.sell_day, df.sell_price, df.inventory_money, df.Tendency],
                                # fill_color='rgb(242,243,243)',
                                fill_color=fill_color,
                                line_color='white',
                                align='center',
                                font=dict(color='black', size=16),
                                height = 35
                                  )
                                )
                        ]
                    )

    fig.update_layout(
                        # width=900,
                        height=600,
                        title = {'text': "<b>Trade Results</b><br>" + 
                                            f"({selector_stock}/USD)", 
                                    'font_family': "Source Sans Pro", 
                                    'font': dict(size=23),
                                    'x' : 0.5
                                    },
                        legend=dict(
                            font = dict(size = 30, color = "white")
                            ),
                     )

    return fig

def plot_trades_equity(df, initial_inventory_money, start_date):

    df_inventory_money = df[['sell_day', 'inventory_money']]
    df_inventory_money = df_inventory_money.set_index('sell_day')

    # Add the first trade in the 1st row, with the initial money

    new_row = pd.DataFrame({'sell_day': start_date,
                            'inventory_money': initial_inventory_money
                           }, index=[0]
                          )
    new_row = new_row.set_index('sell_day')
    df_inventory_money = pd.concat([new_row, df_inventory_money], axis=0, ignore_index=False)

    # Plot the figure

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_inventory_money.index, 
                             y=df_inventory_money['inventory_money'],
                             name='Inventory Money Evolution',
                             line=dict(color='rgb(41, 98, 255)')
                             )
                 )

    fig.update_traces(mode="markers+lines", hovertemplate=None)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        # width=1200,
        # height=375,
        title = {'text': "<b>Current Account Balance</b> ($)", 
                                'font_family': "Source Sans Pro", 
                                'font': dict(size=20),
                                'x': 0
                },
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
    
    # Axes
    fig.update_xaxes(fixedrange = False, showline=True, linewidth=2.5, linecolor='rgb(242,243,243)', gridcolor='rgb(242,243,243)', showspikes=True)
    fig.update_yaxes(fixedrange = False, type="log", showline=True, linewidth=2.5, linecolor='rgb(242,243,243)', gridcolor='rgb(242,243,243)', showspikes=True)

    return fig

def show_prediction_page():

#----------------------------------------------#
# INPUTS

    with open('style.css') as f :
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([2,1,2,2,2,2,1,1,2], gap="small")

    # Selector Stock
    selector_stock = col3.selectbox('Stock', list(stocks.keys()), index=0, format_func=lambda x: stocks[x])
    # Initial Inventory Money

    initial_inventory_money = col4.number_input("Initial Money ($)", min_value=0, value=10000)

    # Selector dates of trading
    selector_start_date = col5.date_input("Start Date", 
                                          value=datetime.strptime("2020-01-01", '%Y-%d-%m'), 
                                          min_value=None, 
                                          max_value=datetime.now().date()
                                         )
    selector_end_date = col6.date_input("End Date", 
                                        max_value=datetime.now().date()
                                       )
    
    try:
        if selector_start_date >= selector_end_date :
            raise DateError
    except DateError:
        st.error('Start date must be less than end date ', icon="🚨")
    else :
    # Substract 30 days, needed to do the predictions
        start_date = selector_start_date - timedelta(days=44)
        # start_date = selector_start_date 

    # Load and Process the data
        # Load the dataset
        df = load_data(selector_stock, str(start_date), str(selector_end_date))

        # Scale Data
        df_scaled, scale_minmax = scale_data(df)

        # Preprocess input data
        data_preprocessed = preprocess_multistep_lstm(df_scaled)

        # Load Model LSTM
        name_model = dict_models[selector_stock]
        model = load_model('Models/' + name_model)

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
        df_historic_trades, logs_trades = simulator(mean_predictions, df, initial_inventory_money)

    #----------------------------------------------#
    # Display Metrics : Currencies, Equity, Balance, Number of Trades

        col1_1, col1_2, col1_3, col1_4, col1_5, col1_6, col1_7, col1_8, col1_9, col1_10, col1_11, col1_12 = st.columns([1,1,1,2,1,2,1,2,1,1,1,1], gap="small")

        balance, profits, profits_percentage, nb_trades, trading_period = calculate_metrics_dashboard(df_historic_trades, initial_inventory_money)

        col1_4.metric("Balance", str(balance) + "$")
        col1_6.metric("Profits", str(profits) + "$", str(profits_percentage) + "%")
        col1_8.metric("Trades", nb_trades, str((trading_period[1]-trading_period[0]).days) + ' trading days')

    #----------------------------------------------#

    # Display charts

        st.set_option('deprecation.showPyplotGlobalUse', False)
        col1_3, col2_3 = st.columns([4,3], gap="small")

        fig = plot_trades(df, df_historic_trades, mean_predictions)
        col1_3.plotly_chart(fig, use_container_width=True, config=dict({'scrollZoom': True}))

    # Display Dataframe (Trades)

        # Calculate the tendency between each buy/sell of each trades
        df_historic_trades = calculate_tendency(df_historic_trades)

        table_trades = plot_table_trades(df_historic_trades, selector_stock)
        col2_3.plotly_chart(table_trades, use_container_width=True)
        

    # Display Equity chart
        col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1], gap="small")

        fig_inventory_money = plot_trades_equity(df_historic_trades, initial_inventory_money, start_date)
        col2.plotly_chart(fig_inventory_money, use_container_width=False, config=dict({'scrollZoom': True}))



























