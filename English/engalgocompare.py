#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:11:07 2017

@author: mohitbindal
"""



# Compare Algorithms
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from utils import ENGLISH_GENRE_LIST,ENGLISH_GENRE_DIR
from sklearn import preprocessing
GENRE_LIST=ENGLISH_GENRE_LIST
GENRE_DIR=ENGLISH_GENRE_DIR
from ceps import read_ceps
# load dataset

X,Y=read_ceps(GENRE_LIST,GENRE_DIR)
print(X)
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
                    """kfold = model_selection.KFold(n_splits=10, random_state=seed)
                    cv_results = model_selection.cross_val_score(model, Z, Y, cv=kfold, scoring=scoring)
                    results.append(cv_results)
                    names.append(name)
                    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                    print(msg)
                    """
                    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=.20, random_state=0)
                    scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train_transformed = scaler.transform(X_train)
                    cv_results = model_selection.cross_val_score(model, X, Y,scoring=scoring)
                    results.append(cv_results)
                    names.append(name)
                    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
