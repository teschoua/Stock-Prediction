import requests
import streamlit as st
import pandas as pd 
from PIL import Image
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import json 
import time
import yfinance as yf
from prediction_page import show_prediction_page

# Page layout
st.set_page_config(layout="wide")


# Title

# image = Image.open('Images/bitcoin.jpg')

# st.image(image, width = 500)

# st.title('Stock Prediction App')
# st.markdown("""
# This app predict the value of a stock market

# """
# )

#----------------------------------------------#
# About
# expander_bar = st.expander("About")
# expander_bar.markdown("""
# * **Python libraries** : pandas, streamlit, numpy, matplotlib
# * **Data Source** : CoinMartketCap
# * **Credit** : ...
# """)

#----------------------------------------------#
# Page layout 
## 3 columns (col1 = sidebar, col2 and col3 = page content)
col1 = st.sidebar
col2, col3 = st.columns((2,1))

#----------------------------------------------#
# Sidebar and Main panel
col1.header('Features')

## Sidebar - Select the page
page = col1.selectbox('Predict Or Explore', ('Predict', 'Explore'))

show_prediction_page()

# if page == 'Predict':
#     show_prediction_page()
# else :
#     show_explore_page()
