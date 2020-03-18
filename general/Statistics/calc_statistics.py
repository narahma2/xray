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
def comparefit(data, fit):    
    yhat = fit
    ybar = np.sum(data) / len(data)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((data - ybar)**2)
    determination = ssreg / sstot
    
    return determination