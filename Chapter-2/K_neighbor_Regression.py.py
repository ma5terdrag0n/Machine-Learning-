# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:38:56 2018

@author: t-prtha
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import scipy as sp
import mglearn

# train_test_split: to split the data into 2 parts
# 1. training_set => 75% data
# 2. testing_set => 25% data
from sklearn.model_selection import train_test_split

# K-Neighbours Regressor
from sklearn.neighbors import KNeighborsRegressor

# Wave dataset from mglearn
X, y = mglearn.datasets.make_wave(n_samples=40)

# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)

# fit the model using the training data and training targets
reg.fit(X_train, y_train)

# Prediction On test data
print("Test set predictions:\n{}".format(reg.predict(X_test)))

# We can also evaluate the model using the score method, 
# which for regressors returns the R^2 score. The R^2 score, 
# also known as the coefficient of determination, 
# is a measure of goodness of a prediction for a regression model, 
# and yields a score between 0 and 1.
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))
