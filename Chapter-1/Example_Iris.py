# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

# Import Required Packages
import numpy as np
import scipy as sp
import sklearn
import pandas as pd
from sklearn.datasets import load_iris
# train_test_split: to split the data into 2 parts
# 1. training_set => 75% data
# 2. testing_set => 25% data
from sklearn.model_selection import train_test_split
# K-nearest neighbour
from sklearn.neighbors import KNeighborsClassifier

# Loading data to a variable
# Data is in the form of dictionary with key and value pairs
iris_dataset = load_iris()
print(iris_dataset.keys())

# print the description of iris dataset
print(iris_dataset['DESCR'] + '\n')

# print target names of dataset
print(iris_dataset['target_names'])

# print feature names
print(iris_dataset['feature_names'])

# Type of iris_dataset['data'] = numpy.ndarray
print("Type of data: {}".format(type(iris_dataset['data'])))

X_train, X_test, y_train, y_test = \
    train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# We can use graphs - (matplot-lib , pandas) to visualize our problem and to see whether our
# problem can be solved using machine learning or can be solved using simple conditions.
# Use pair plot if the dimension of data or number of features are more than 3

# K- nearest model
knn = KNeighborsClassifier(n_neighbors=1)
# fit function builds the learning model
knn.fit(X_train, y_train)

# Making predictions
X_new = np.array([[5, 2.9, 1, 0.2]])
solution = knn.predict(X_new)
print(solution)

# Predicting the result of our test data created when we split the data
y_pred = knn.predict(X_test)

# Checking the score
print(knn.score(X_test, y_test))
# or
print(np.mean(y_test == y_pred))    
