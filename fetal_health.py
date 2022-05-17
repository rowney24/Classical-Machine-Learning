# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:57:19 2022

@author: Ronald Chitauro
"""


#0th step: loading important packages
#basics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  #for plotting the heat map

#Machine Learning imports: Will use Random Forest for feature selection
from sklearn.model_selection import train_test_split # to split the data into a training and test set
#For using Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score



#for all columns of the csv file to be displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#1st step: reading the data
fetalDF = pd.read_csv("fetal_health.csv")
fetalDF.head().T #View a few records of the data in a transposed manner

#2nd Step: Checking if our dataset has any missing data AND just checking simple statistics
fetalDF.info()
fetalDF.isnull().sum()  #This shows that there is no missing data

#Just a quick glance of what our data looks like
fetalDF.describe(include='all').T

#4th step: Splitting the data into test and training set
y = fetalDF['fetal_health']
X = fetalDF.drop("fetal_health", axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


#5th step: training the model
rf_fetus = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
rf_fetus.fit(X_train,y_train)

#Checking the most important feautures


plt.bar(X.columns, rf_fetus.feature_importances_)
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees

#Selecting the most important features
sfm_fetus = SelectFromModel(rf_fetus, threshold=0.05) #Threshold selected according to the coefficients of the most important features

#Train the selector
sfm_fetus.fit(X_train, y_train)

#Create A DataFrame with the most important features
X_important_train = sfm_fetus.transform(X_train)
X_important_test = sfm_fetus.transform(X_test)


# Training a random forest with the most important features

fetus_important_rf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
fetus_important_rf.fit(X_important_train, y_train)






