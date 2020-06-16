"""
-*- coding: utf-8 -*-
Summarize the horizontal correction factors for the jets.

@Author: rahmann
@Date:   2020-05-28
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
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from general.Statistics.calc_statistics import polyfit
from general.misc import create_folder


def get_xpos(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    xpos = np.nanmean(processed_data['Lateral Position'])

    return xpos


def get_mean_elpsT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    mean_elpsT = np.nanmean(processed_data['Transmission Ratios'][0])

    return mean_elpsT


def get_mean_peakT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    mean_peakT = np.nanmean(processed_data['Transmission Ratios'][1])

    return mean_peakT


def main():
    # Location of APS 2018-1 data
    prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'

    # Save location for the plots
    plots_folder = create_folder('{0}/Figures/Jet_HorizSumm/'.format(prj_fld))

    # Scintillator
    scintillators = ['LuAG', 'YAG']

    # KI %
    KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]

    # Test matrix
    test_matrix = pd.read_csv(
                              '{0}/APS White Beam.txt'.format(prj_fld),
                              sep='\t+',
                              engine='python'
                              )

    # Crop down the test matrix
    test_matrix = test_matrix[['Test', 'Nozzle Diameter (um)', 'KI %']].copy()

    for scint in scintillators:
        # Processed data sets location
        prc_fld = '{0}/Processed/{1}/Summary/'.format(prj_fld, scint)

        # Groups
        rpkT = 'Ratio Peak T'
        relT = 'Ratio Ellipse T'

        # Horizontal variation
        horiz_matrix = test_matrix[test_matrix['Test'].str\
                                                      .contains('mm')]\
                                                      .copy()
        horiz_matrix['X Position'] = [
                                      get_xpos('{0}/{1}_{2}.pckl'\
                                               .format(prc_fld, scint, x))
                                      for x in horiz_matrix['Test']
                                      ]

        # Sort horiz_matrix by X position, re-index, and drop the outliers
        horiz_matrix.sort_values(by=['X Position'], inplace=True)
        horiz_matrix.reset_index(inplace=True)
        horiz_matrix.drop([0, len(horiz_matrix)-1], inplace=True)

        # Get horizontal values
        horiz_matrix[relT] = [
                              get_mean_elpsT('{0}/{1}_{2}.pckl'\
                                             .format(prc_fld, scint, x))
                              for x in horiz_matrix['Test']
                              ]
        horiz_matrix[rpkT] = [
                              get_mean_peakT('{0}/{1}_{2}.pckl'\
                                             .format(prc_fld, scint, x))
                              for x in horiz_matrix['Test']
                              ]
        breakpoint()
        # Create a fit to the horizontal variation
        xData = horiz_matrix['X Position']
        yData = horiz_matrix[relT]
        yData = savgol_filter(yData, 55, 3)
        XtoCF = CubicSpline(xData, yData)

        # Horizontal plot
        plt.figure()
        plt.plot(
                 horiz_matrix['X Position'],
                 horiz_matrix[rpkT],
                 color='olivedrab',
                 marker='s',
                 label=rpkT
                 )
        plt.plot(
                 xData,
                 XtoCF(xData),
                 color='cornflowerblue',
                 label='Fit'
                 )
        plt.plot(
                 horiz_matrix['X Position'],
                 horiz_matrix[relT],
                 fillstyle='none',
                 color='olivedrab',
                 marker='s',
                 label=relT
                 )
        plt.legend()
        plt.title('{0} - Horizontal Variation - 700 um, 10% KI'.format(scint))
        plt.savefig('{0}/{1}_horiz.png'.format(plots_folder, scint))
        plt.close()

        # Save the linear fitted correction factors
        with open('{0}/Processed/{1}/{1}_peakT_cf.txt'\
                  .format(prj_fld, scint), 'wb') as f:
            np.savetxt(f, peakT_combi_fit['function'](KI_conc))

        with open('{0}/Processed/{1}/{1}_elpsT_cf.txt'\
                  .format(prj_fld, scint), 'wb') as f:
            np.savetxt(f, elpsT_combi_fit['function'](KI_conc))


if __name__ == '__main__':
    main()
