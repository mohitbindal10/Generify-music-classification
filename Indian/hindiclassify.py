#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:24:24 2017

@author: mohitbindal
"""

import os
import timeit
import numpy as np
from collections import defaultdict
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.metrics import precision_recall_curve, roc_curve,recall_score
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics.scorer import make_scorer
from utils import GENRE_LIST, GENRE_DIR, TEST_DIR
from utils import plot_confusion_matrix, plot_roc_curves
from ceps import read_ceps, read_ceps_test
genre_list = GENRE_LIST
start = timeit.default_timer()
print("\n")
print (" Starting classification \n")
print (" Classification running ... \n") 
X, y = read_ceps(genre_list,GENRE_DIR)
print(" X is " , X, "len of x is ",len(X),X.shape)
print("y is ",y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.20, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = KNN().fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
print(clf.score(X_test_transformed, y_test))
predicted = cross_val_predict(clf, X, y, cv=10)
joblib.dump(clf, 'saved_model/hindiknnmodell.pkl')

print(metrics.accuracy_score(y, predicted) )
scoring = {'prec_macro': 'precision_macro','rec_micro': make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, X, y, scoring=scoring,cv=5, return_train_score=True)
sorted(scores.keys())                 
print(scores['train_rec_micro'] )