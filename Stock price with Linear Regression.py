# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 07:54:45 2020

@author: Mrinmoi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

stock_prices=pd.read_csv("D:/Machine Learning/Python+for+Financial+Analysis+-+Course+Package/Part 3. AI and ML in Finance/stock.csv")
stock_volume=pd.read_csv('D:/Machine Learning/Python+for+Financial+Analysis+-+Course+Package/Part 3. AI and ML in Finance/stock_volume.csv')
stock_prices=stock_prices.sort_values(by=['Date'])
stock_volume=stock_volume.sort_values(by=['Date'])
stock_prices.isnull().sum()
stock_volume.isnull().sum()

def Individual_stock(price,volume,name):
    return pd.DataFrame({'Date':stock_prices['Date'],'price':stock_prices[name],'volume':stock_volume[name]})
def Trading_window(data):
    n=1
    data['Target']=data['price'].shift(-n)
    return data
AAPL=Individual_stock(stock_prices, stock_volume, 'AAPL')
AAPL
AAPL_target=Trading_window(AAPL)
AAPL_target=AAPL_target[:-1]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
AAPL_target_scaled=sc.fit_transform(AAPL_target.drop(columns=['Date']))
x=AAPL_target_scaled[:,:2]
y=AAPL_target_scaled[:,2:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=7)

from sklearn.linear_model import Ridge
regression_model=Ridge(alpha=4)
regression_model.fit(x_train,y_train)
AAPL_predictions=regression_model.predict(x_test)
accuracy=regression_model.score(x_test,y_test)
accuracy

predicted=[]
for i in AAPL_predictions:
    predicted.append(i[0])

original=[]
for i in y_test:
    original.append(i[0])

AAPL_prediction_dataframe=pd.DataFrame({'original':original,'prediction':predicted})

