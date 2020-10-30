# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:43:12 2020

@author: Mrinmoi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("D:/ML/Datasets/data.csv")
data

data.columns
data.dtypes
data.isnull().sum()
data=data.drop(['id','Unnamed: 32'],axis=1)
data

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['diagnosis']=le.fit_transform(data['diagnosis'])

#visualization

sns.pairplot(data, hue='diagnosis' ,vars=['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean'])
sns.countplot(data['diagnosis'])
sns.scatterplot(x='radius_mean',y='texture_mean',hue='diagnosis',data=data)
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True)

x=data.drop('diagnosis', axis=1)
y=data['diagnosis']
y

#split data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=7)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)



#train and evaluate model

from sklearn.svm import SVC
svc_model=SVC()
svc_model.fit(x_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100, random_state=7)
rfc.fit(x_train,y_train)


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
y_pred=svc_model.predict(x_test)
y_pred2=rfc.predict(x_test)
cm=confusion_matrix(y_test,y_pred2)
cm
af=accuracy_score(y_test,y_pred2)
af
cr=classification_report(y_test,y_pred2)
cr
sns.heatmap(cm, annot=True)

#improving the model(if necessary)

param={'C':[0.1,1,10,100],'gamma':[0.1,1,10,100],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid1=GridSearchCV(SVC(),param,refit=True,verbose=4)
grid1.fit(x_train,y_train)
grid1.best_params_
grid1_pred=grid1.predict(x_test)
cm2=confusion_matrix(y_test,grid1_pred)
sns.heatmap(cm2, annot=True)
af2=accuracy_score(y_test,grid1_pred)
af2

