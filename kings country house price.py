# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:57:37 2020

@author: Mrinmoi
"""
pip install xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("D:/ML/Datasets/kc_house_data.csv")
data.isnull().sum()
data.describe()
data=data.drop(["id","date","yr_built","yr_renovated",],axis=1)
x=data.iloc[:,1:].values
y=data.iloc[:,0:1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

from xgboost import xgclassifier

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

from sklearn.ensemble import RandomForestRegressor
reg2 = RandomForestRegressor(n_estimators = 100,  random_state = 1)
reg2.fit(x_train, y_train)

y_pred2 = reg2.predict(x_test)
y_pred=reg.predict(x_test)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print(mean_absolute_error(y_test,y_pred2)) 
print(y_pred2,y_test)

X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

