#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:48:03 2020

@author: hamzah
"""

#import lib
from imblearn.under_sampling import TomekLinks, ClusterCentroids
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
import imblearn
from sklearn.preprocessing import LabelEncoder
import collections
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

## import data file
wine = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    , delimiter=";")
print("-------------")
print("---COLUMNS----")
print(wine.columns)
print("-------------")

plt.figure(figsize=(10, 6))
sns.countplot(wine["quality"], palette="muted")
print("-----PLOT COUNT---")

print("---VALUE COUNT----")
print(wine["quality"].value_counts())
print("-------------")



quality = wine["quality"].values
category = []
for num in quality:
    if num < 5:
        category.append("Low")
    elif num > 6:
        category.append("High")
    else:
        category.append("Medium")
       
print("Categories and count---->")
print([(i, category.count(i)) for i in set(category)])
##barplot of the response after transformation¶
print("##barplot of the response after transformation¶")
plt.figure(figsize=(10, 6))
sns.countplot(category, palette="muted")


##Cor Mat to check features¶
print("##Cor Mat to check features¶")
plt.figure(figsize=(12, 6))
sns.heatmap(wine.corr(), annot=True)



###Set up model matrix¶
print("###Set up model matrix¶")
quality = wine["quality"].values
category = []
for num in quality:
    if num < 5:
        category.append("Low")
    elif num > 6:
        category.append("High")
    else:
        category.append("Midium")
category = pd.DataFrame(data=category, columns=["category"])
data = pd.concat([wine, category], axis=1)
data.drop(columns="quality", axis=1, inplace=True)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2018)


#random forest
print("#random forest")
clf = RandomForestClassifier(random_state=2018, oob_score=True)
param_dist = {"n_estimators": [50, 100, 150, 200, 250],
              'min_samples_leaf': [1, 2, 4]}
rfc_gs = GridSearchCV(clf, param_grid=param_dist, scoring='accuracy', cv=5)
rfc_gs.fit(X_train, y_train)

print("---------------")
print("Best Score")
print(rfc_gs.best_score_)
print("---------------")


#Decision Tree
clf = Pipeline([
    ('scl', StandardScaler()),
    ('pca', PCA(random_state=42)),
    ('clf', DecisionTreeClassifier(random_state=42))])

print("---------------")
print("###Decision Tree")
print(clf)
print("---------------")

criterion = ['gini', 'entropy']
print("------->Criterion:".join(criterion))
splitter = ['best']
print("------->splitter:".join(splitter))
max_depth = [8, 9, 10, 11, 15, 20, 25]
print("------->max_depth:")
print(max_depth)
min_samples_leaf = [2, 3, 5]
print("------->min_samples_leaf:")
print(min_samples_leaf)
class_weight = ['balanced', None]
print("------->class_weight:")
print(class_weight)

print("................")

param_grid =\
    [{'clf__class_weight': class_weight,
      'clf__criterion': criterion,
      'clf__splitter': splitter,
      'clf__max_depth': max_depth,
      'clf__min_samples_leaf': min_samples_leaf
      }]
print("---------------")
print("Param grid")
print(param_grid)
print("---------------")

gs_dt = GridSearchCV(estimator=clf, param_grid=param_grid,
                     scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_dt.fit(X_train, y_train)

print("------GridSearch CV---------")
print(gs_dt)
print("---------------")

print("--Best Score--")
print(gs_dt.best_score_)
print("---------------")


# under sample "2" ；
# over sample "1", "0"
smt = ClusterCentroids(ratio={2: 1500})
X_sm, y_sm = smt.fit_sample(X_train, y_train)
smt2 = SMOTE(ratio={0: 1500, 1: 1500})
X_sm2, y_sm2 = smt2.fit_sample(X_sm, y_sm)

#Random Forest
print("#Random Forest ")
rfc_rs = RandomForestClassifier(random_state=2018)
param_dist = {"n_estimators": [50, 100, 150, 200, 250],
              'min_samples_leaf': [1, 2, 4]}
rfc_gs_rs = GridSearchCV(rfc_rs, param_grid=param_dist,
                         scoring='accuracy', cv=5)
rfc_gs_rs.fit(X_sm2, y_sm2)

print("New best Score")
rfc_gs_rs.best_score_
print(rfc_gs_rs.best_score_)

importances = rfc_gs_rs.best_estimator_.feature_importances_

print("--importances--")
print(importances)
print("---------------")

wine.columns[:-1]

feature_importances = pd.DataFrame(importances,index = wine.columns[:-1],
                                    columns=['importance']).sort_values('importance',
                                                                        ascending=False)

print("--feature_importances--")
print(feature_importances)
print("---------------")

print("Creating Horizontal bar chat--")
feature_importances.plot(kind='barh')

print("PLOT new graph")



print("==================END======================")



