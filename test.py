# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:58:58 2023

@author: Raghav Goel
"""
import numpy as np
import LinearRegression
import matplotlib.pyplot

data = np.genfromtxt("50Startups_Train.csv", delimiter=",", skip_header=1, usecols=(0, 1, 2, 4))
X = data[:,0:3]
y = data[:, 3]

test_data = np.genfromtxt("50Startups_Test.csv", delimiter=",", skip_header=1, usecols=(0, 1, 2, 4))
X_test = test_data[:,0:3]
y_test = test_data[:, 3]

lr = LinearRegression.Model(X, y, scale=True)
iters = lr.fit(max_iter=10000)
W = lr.W
Y_pred = lr.predict(X_test)