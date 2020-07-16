# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:03:54 2018

@author: Naveed
"""

import numpy as np

# Polynomial regression
def polyfit(x, y, degree):
    results = {}
    
    coeffs = np.polyfit(x, y, degree)
    
    # Polynomial coefficients
    results['polynomial'] = coeffs.tolist()
    
    # r-squared
    p = np.poly1d(coeffs)
    results['function'] = p
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['determination'] = ssreg / sstot
    
    return results

# Compare fit to data
def comparefit(observed_data, model):
    # Equations from <https://en.wikipedia.org/wiki/Coefficient_of_determination>    
    ybar = np.sum(observed_data) / len(observed_data)
#    ssreg = np.sum((model - ybar)**2)
    ssres = np.sum((observed_data - model)**2)
    sstot = np.sum((observed_data - ybar)**2)
    determination = 1 - (ssres / sstot)
    
    mse = ssres / len(observed_data)
    rmse = np.sqrt(mse)
    
    return determination, rmse