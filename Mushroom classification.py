# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:07:32 2020

@author: Mrinmoi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#dataset

data=pd.read_csv("D:/Machine Learning/Datasets/478_974_bundle_archive/mushrooms.csv")
data
data.describe()
data.columns.unique()
data.isnull().sum()
data['cap-shape'].unique()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['cap-shape']=le.fit_transform(data['cap-shape'])
data['cap-surface']=le.fit_transform(data['cap-surface'])
data['cap-color']=le.fit_transform(data['cap-color'])
data['bruises']=le.fit_transform(data['bruises'])
data['odor']=le.fit_transform(data['odor'])
data['gill-attachment']=le.fit_transform(data['gill-attachment'])
data['gill-spacing']=le.fit_transform(data['gill-spacing'])
data['gill-size']=le.fit_transform(data['gill-size'])
data['gill-color']=le.fit_transform(data['gill-color'])
data['stalk-shape']=le.fit_transform(data['stalk-shape'])
data['stalk-root']=le.fit_transform(data['stalk-root'])
data['stalk-surface-above-ring']=le.fit_transform(data['stalk-surface-above-ring'])
data['stalk-surface-below-ring']=le.fit_transform(data['stalk-surface-below-ring'])
data['stalk-color-above-ring']=le.fit_transform(data['stalk-color-above-ring'])
data['stalk-color-below-ring']=le.fit_transform(data['stalk-color-below-ring'])
data['veil-type']=le.fit_transform(data['veil-type'])
data['veil-color']=le.fit_transform(data['veil-color'])
data['ring-number']=le.fit_transform(data['ring-number'])
data['ring-type']=le.fit_transform(data['ring-type'])
data['spore-print-color']=le.fit_transform(data['spore-print-color'])
data['population']=le.fit_transform(data['population'])
data['habitat']=le.fit_transform(data['habitat'])
data['class']=le.fit_transform(data['class'])

x=data.drop(['class'],axis=1)
y=data['class']

#visualization

sns.countplot(data['cap-shape'])
sns.pairplot(data,hue='class',vars=['cap-shape','cap-color','bruises','odor'])
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot=True)

#train_test_split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=7)

#training model

from sklearn.svm import SVC
svc_model=SVC()
svc_model.fit(x_train,y_train)

from sklearn.tree import DecisionTreeClassifier
dec=DecisionTreeClassifier()
dec.fit(x_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier(n_estimators=10,random_state=7)
rc.fit(x_train,y_train)

#evaluating model

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

svc_pred=svc_model.predict(x_test)
cm_svc=confusion_matrix(y_test,svc_pred)
cm_svc
af_svc=accuracy_score(y_test,svc_pred)
af_svc

dec_pred=dec.predict(x_test)
cm_dec=confusion_matrix(y_test,dec_pred)
cm_dec
af_dec=accuracy_score(y_test,dec_pred)
af_dec

rc_pred=rc.predict(x_test)
cm_rc=confusion_matrix(y_test,rc_pred)
cm_rc
af_rc=accuracy_score(y_test,rc_pred)
af_rc

sns.heatmap(cm_svc,annot=True)
sns.heatmap(cm_dec,annot=True)
sns.heatmap(cm_rc,annot=True)
