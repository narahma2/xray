"""
-*- coding: utf-8 -*-
Summarize the errors for the calibration set.

@Author: rahmann
@Date:   2020-05-29
@Last Modified by:   rahmann
"""

import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from skimage.transform import rotate
from general.calc_statistics import polyfit
from general.misc import create_folder


def get_ypos(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    ypos = processed_data['Axial Position']

    return ypos


def get_abs(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    if method == 'Ellipse':
        diam = 1
    elif method == 'Peak':
        diam = 2

    # Return absolute error in um
    abs_err = 10000 * (np.array(processed_data['Diameters'][diam]) -
                       np.array(processed_data['Diameters'][0]))

    return abs_err


def reject_outliers(vert_loc, data, m=2):
    ind = abs(data - np.nanmedian(data)) < m * np.nanstd(data)

    return vert_loc[ind], data[ind]

def get_rmse(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return RMSE in um
    rmse = 10000 * np.array(processed_data[method + ' Errors'][0])

    return rmse


def get_mape(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return MAPE/mean absolute percentage error (already scaled to 100)
    mape = np.array(processed_data[method + ' Errors'][1])

    return mape


def get_zeta(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return median symmetric accuracy as percentage (already scaled to 100)
    zeta = np.array(processed_data[method + ' Errors'][2])

    return zeta


def get_mdlq(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return MdLQ/median log accuracy ratio (no units)
    # Positive/negative: systematic (over/under)-prediction
    mdlq = np.array(processed_data[method + ' Errors'][3])

    return mdlq


def main():
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'
    cor_fld = '{0}/Corrected/YAG/Summary/'.format(prj_fld)
    plt_fld = create_folder('{0}/Figures/Cal_Errors/'.format(prj_fld))

    tests = glob.glob('{0}/*.pckl'.format(cor_fld))

    # Container for what will be the summary DataFrame
    d = []

    for test in tests:
        # Use 'Ellipse' as the method (worked best in 2018-1)
        method = 'Ellipse'

        # Retrieve test name
        test_name = test.rsplit('/')[-1].rsplit('.')[0]

        # Retrieve vertical locations
        vert_loc = np.array(get_ypos(test))

        # Retrieve errors
        abEr = get_abs(test, method)
        rmse = get_rmse(test, method)
        mape = get_mape(test, method)
        zeta = get_zeta(test, method)
        mdlq = get_mdlq(test, method)

        # Remove outliers from data
        vert_loc, abEr = reject_outliers(vert_loc, abEr)

        # Build up summary dict
        d.append({
                  'Name': test_name,
                  'RMSE': rmse,
                  'MAPE': mape,
                  'Zeta': zeta,
                  'MdLQ': mdlq
                  })

        # Create absolute error plot
        plt.figure()
        plt.plot(vert_loc, abEr, color='tab:blue', linewidth=2.0)
        plt.xlabel('Vertical Location (px)')
        plt.ylabel('Absolute Error ($\mu$m)')
        plt.title(test_name)
        plt.savefig('{0}/{1}.png'.format(plt_fld, test_name))
        plt.close()

    # Save summary of errors
    summ = pd.DataFrame(d)
    summ.to_csv(prj_fld + '/Corrected/YAG/summary.txt', sep='\t')


# Run this script
if __name__ == '__main__':
    main()
