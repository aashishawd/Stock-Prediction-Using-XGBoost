# Stock Market Analysis And Price Prediction
The S&P BSE SENSEX is a free-float market-weighted stock market index of 30 well-established and financially sound companies listed on Bombay Stock Exchange. In this project, I shall analyze historical S&P BSE Sensex data, particularly the Open, High, Low and Close over the past 10 years. I shall then calculate various technical indicators used in market analysis to forecast BSE market performance, using the famous XGBoost regressor. Then, I shall do the same using LSTMs, compare the outcomes and finalize my model.

## Overview

This project has 5 steps:

1. Data Preprocessing </br>
This dataset is clean and requires no preprocessing.
```
df = pd.read_csv(Location of dataset, index_col=False)
dfp=df
df['Date']=df['Date'].astype('datetime64')
df.head()
```

2. Data Visualization </br>
Candlestick charts are the language of stocks. Also, since we will be using time series forecasting, it only makes sense to decompose the data and evaluate trends and seasonality. The code can be found in dataviz.py.

3. Plotting Technical Indicators </br>
Technical indicators are the stock trader's toolbox. I start off by plotting exponential and simple moving averages over different time periods. Then, I plot the RSI, which is an indicator of whether the stocks of the index are oversold or overbought. Finally, I plot the MACD to observe the market momentum. The code can be found in techind.py.

4. Using XGBoost to predict closing prices </br>
Not all indicators are equally important. Understanding the weightage alloted to each one allows us to make optimal decisions. We shall use GridSearch to find the parameters with the best validation score and use those in our model. The code can be found in xgb.py.

5. Using LSTMs for time series forecasting </br>
Our model doesn't do a very good job of predictions. Also, it relies on calculation of a lot of parameters and is computationally expensive. A better option would be to use LSTMs for predicting the future closing prices. The code can be found in lst.py.

## Data

BSE allows you to download historical data (Open, High, Low, Close) of the indices. You can find the link to do so here: https://www.bseindia.com/indices/IndexArchiveData.html

## Tools Used

* Python 3.6
* Numpy
* Pandas
* Xgboost
* Sklearn
* Plotly
* Stldecompose

```
#importing relevant libraries
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import types
from botocore.client import Config
import ibm_boto3

# Plotting    
!pip install plotly==4.6.0
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Time series decomposition
!pip install stldecompose
from stldecompose import decompose
```
