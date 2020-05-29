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
from general.Statistics.calc_statistics import polyfit
from general.misc import create_folder


def get_ypos(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    ypos = processed_data['Axial Position']

    return ypos


def get_rmse(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return RMSE in um
    rmse = 10000 * np.array(processed_data[method + 'Errors'][0])

    return rmse


def get_mape(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return MAPE/mean absolute percentage error (already scaled to 100)
    mape = np.array(processed_data[method + 'Errors'][1])

    return mape


def get_zeta(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return median symmetric accuracy as percentage (already scaled to 100)
    zeta = np.array(processed_data[method + 'Errors'][2])

    return zeta


def get_mdlq(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return MdLQ/median log accuracy ratio (no units)
    # Positive/negative: systematic (over/under)-prediction
    mdlq = np.array(processed_data[method + 'Errors'][3])

    return mdlq


def main():
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'
    cor_fld = '{0}/Corrected/YAG/Summary/'.format(prj_fld)
    plt_fld = create_folder('{0}/Figures/Cal_Errors/'.format(prj_fld))
    rmse_fld = create_folder('{0}/RMSE/'.format(plt_fld))
    mape_fld = create_folder('{0}/MAPE/'.format(plt_fld))
    zeta_fld = create_folder('{0}/zeta/'.format(plt_fld))
    mdlq_fld = create_folder('{0}/MdLQ/'.format(plt_fld))

    tests = glob.glob('{0}/*.pckl'.format(cor_fld))

    for test in tests:
        # Use 'Ellipse' as the method (worked best in 2018-1)
        method = 'Ellipse'

        # Retrieve test name
        test_name = test.rsplit('/')[-1].rsplit('.')[0]

        # Retrieve vertical locations
        vert_loc = get_ypos(test)

        # Retrieve errors
        rmse = get_rmse(test, method)
        mape = get_mape(test, method)
        zeta = get_zeta(test, method)
        mdlq = get_mdlq(test, method)

        # Create RMSE plot
        plt.figure()
        plt.plot(vert_loc, rmse, color='tab:blue', linewidth=2.0)
        plt.xlabel('Vertical Location (px)')
        plt.ylabel('RMSE ($\mu$m)')
        plt.title(test_name + ': RMSE')
        plt.savefig('{0}/{1}.png'.format(rmse_fld, test_name))
        plt.close()

        # Create MAPE plot
        plt.figure()
        plt.plot(vert_loc, mape, color='tab:orange', linewidth=2.0)
        plt.xlabel('Vertical Location (px)')
        plt.ylabel('MAPE (%)')
        plt.ylim([0, 100])
        plt.title(test_name + ': MAPE')
        plt.savefig('{0}/{1}.png'.format(mape_fld, test_name))
        plt.close()

        # Create zeta plot
        plt.figure()
        plt.plot(vert_loc, zeta, color='tab:green', linewidth=2.0)
        plt.xlabel('Vertical Location (px)')
        plt.ylabel('$\zeta$ (%)')
        plt.ylim([0, 100])
        plt.title(test_name + ': $\zeta$')
        plt.savefig('{0}/{1}.png'.format(zeta_fld, test_name))
        plt.close()

        # Create MdLQ plot
        plt.figure()
        plt.plot(vert_loc, mdlq, color='tab:red', linewidth=2.0)
        plt.xlabel('Vertical Location (px)')
        plt.ylabel('MdLQ (-)')
        plt.title(test_name + ': MdLQ')
        plt.savefig('{0}/{1}.png'.format(mdlq_fld, test_name))
        plt.close()


# Run this script
if __name__ == '__main__':
    main()
