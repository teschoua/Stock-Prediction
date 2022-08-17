import requests
import streamlit as st
import pandas as pd 
from PIL import Image
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bs4 import BeautifulSoup
import requests
import json 
import time
import yfinance as yf

# Page layout
st.set_page_config(layout="wide")


# Title

# image = Image.open('Images/bitcoin.jpg')

# st.image(image, width = 500)

st.title('Stock Prediction App')
st.markdown("""
This app predict the value of a stock market

"""
)

#----------------------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries** : base64, p andas, streamlit, numpy, matplotlib
* **Data Source** : CoinMartketCap
* **Credit** : ...
""")

#----------------------------------------------#
# Page layout 
## 3 columns (col1 = sidebar, col2 and col3 = page content)
col1 = st.sidebar
col2, col3 = st.columns((2,1))

#----------------------------------------------#
# Sidebar and Main panel
col1.header('Input Options')

## Sidebar - Currency price unit
# currency_price_unit = col1.selectbox('Select currency for price', ('USD', 'BTC', 'ETH'))

currency_price_unit = col2.selectbox('Select currency for price', ('None', 'GOOGL', 'AAPL', 'MSFT', 'AMZN'))

@st.cache
def load_data():
    
    tickerSymbols = "GOOGL AAPL MSFT AMZN"

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

df = load_data()