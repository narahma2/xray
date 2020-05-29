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


# Calculate RMSE (root mean squared error)
def rmse(experimental, theoretical):
    """Root-mean-square error. Same units as data."""
    experimental = np.array(experimental)
    theoretical = np.array(theoretical)
    rmse = np.sqrt(np.nanmean((experimental - theoretical)**2))

    return rmse


# Calculate MAPE (mean absolute percentage error)
def mape(experimental, theoretical):
    """Mean absolute percentage error. Units are percentage scaled to 100."""
    experimental = np.array(experimental)
    theoretical = np.array(theoretical)
    mape = 100*np.sum(np.abs((experimental - theoretical)/theoretical)) / len(theoretical)

    return mape


# Calculate median symmetric accuracy
def zeta(experimental, theoretical):
    """Median symmetric accuracy. Units are percentage scaled to 100."""
    exp = np.array(experimental)
    theo = np.array(theoretical)
    mdlq = calc_mdlq(
    zeta = 100*(np.exp(
                       np.median(
                                 np.abs(
                                        np.log(exp/theo)
                                        )
                                 )
                       ) - 1
                )

    return zeta


# Calculate MdLQ (median log accuracy ratio)
# Positive/negative: systematic (over/under)-prediction
def mdlq(experimental, theoretical):
    """Median log accuracy ratio. Unit-less."""
    experimental = np.array(experimental)
    theoretical = np.array(theoretical)
    mdlq = np.median(np.log(experimental/theoretical))

    return mdlq


