# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:58:58 2023

@author: Raghav Goel
"""
import numpy as np
import LinearRegression

x = np.arange(10.0).reshape(10, 1)
m = 3
b = 9
y = (x * m + b).reshape(-1, )
w = np.zeros(x.shape[-1] + 1)

lr = LinearRegression.Model(x, w, y) 
iterations = lr.fit(10000)

print(lr.W)
print(iterations)