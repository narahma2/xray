"""
-*- coding: utf-8 -*-
Summarize the errors for the corrected jets.

@Author: rahmann
@Date:   2020-05-02 19:55:00
@Last Modified by:   rahmann
"""

import sys
if sys.platform == 'win32':
    sys.path.append('E:/GitHub/xray/general')
    sys.path.append('E:/GitHub/xray/temperature')
    sys_folder = 'R:'
elif sys.platform == 'linux':
    sys.path.append('/mnt/e/GitHub/xray/general')
    sys.path.append('/mnt/e/GitHub/xray/temperature')
    sys_folder = '/mnt/r/'

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
from Statistics.calc_statistics import polyfit
from jet_processing import create_folder
from timeit import default_timer as timer

# Location of APS 2018-1 data
prj_fld = '{0}/X-ray Radiography/APS 2018-1/'.format(sys_folder)

# Save location for the plots
plt_fld = create_folder('{0}/Figures/Jet_Errors/'.format(prj_fld))

# KI %
KIconc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]

def get_xpos(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    xpos = np.nanmean(processed_data['Lateral Position'])

    return xpos


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


def get_rmse(path, method):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    # Return RMSE in um
    rmse = 10000 * np.array(processed_data[method + ' Errors'][0])

    return rmse


def vert_var(vert_matrix, method, corr_pckl, scint):
    # Create new column
    clmn = method + ' AbsErr'
    vert_matrix[clmn] = np.nan

    abs_errors = []
    for i,x in enumerate(vert_matrix['Test']):
        pckl_file = '{0}{1}.pckl'.format(corr_pckl, x) \
                                 .replace('XYZ', method.lower())
        abs_errors.append(get_abs(pckl_file, method))

    # Assign values to column
    vert_matrix[clmn] = abs_errors

    # Group the matrix by diameter and KI%
    vert_MAE_grouped = vert_matrix.copy()
    grp1 = 'Nozzle Diameter (um)'
    grp2 = 'KI %'
    vert_MAE_grouped = vert_MAE_grouped.groupby([grp1, grp2]) \
                                       .apply(lambda x: np.nanmean(x[clmn],
                                                                   axis=0))

    # Get axial positions
    axial_positions = np.linspace(start=20, stop=325, num=325-20+1, dtype=int)
    axial_positions = axial_positions[2:-1]

    linecolors = ['dimgray', 'firebrick', 'goldenrod', 'mediumseagreen',
                  'steelblue', 'mediumpurple', 'hotpink']
    linelabels = ['0%', '1.6%', '3.4%', '4.8%', '8%', '10%', '11.1%']

    warnings.filterwarnings('ignore')
    # Vertical absolute error plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(12, 6)

    # Plot the 700 um jets
    for i,x in enumerate(linecolors):
        KI = KIconc[i]
        ax1.plot(axial_positions, vert_MAE_grouped[700, KI][2:-1],
                 color=x, linewidth=2.0)

    # Plot the 2000 um jets
    for i,x in enumerate(linecolors):
        KI = KIconc[i]
        ax2.plot(axial_positions, vert_MAE_grouped[2000, KI][2:-1],
                 color=x, linewidth=2.0)

    ax1.title.set_text(r'700 $\mu$m')
    ax2.title.set_text(r'2000 $\mu$m')
    ax1.set_xlabel('Axial Position (px)')
    ax2.set_xlabel('Axial Position (px)')
    ax1.set_ylabel('Mean Absolute Error ($\mu$m)')
    fig.suptitle('Vertical Variation - ' + method + ' MAE')
    fig.legend(ax1, labels=linelabels, loc='center right',
               borderaxespad=0.1, title='KI %')
    plt.subplots_adjust(wspace = 0.05, top = 0.90)
    plt.savefig('{0}/{1}_vert_{2}MAE.png'.format(plt_fld, scint, method))
    plt.close()
    warnings.filterwarnings('default')

    return vert_matrix


def horiz_var(horiz_matrix, method, corr_pckl):
    # Sort horiz_matrix by X position, reset index, and drop the edges
    horiz_matrix.sort_values(by=['X Position'], inplace=True)
    horiz_matrix.reset_index(inplace=True)
    horiz_matrix.drop([0, len(horiz_matrix)-1], inplace=True)

    # Get horizontal values
    horiz_matrix[method + ' RMSE'] = np.nan

    for i,x in enumerate(horiz_matrix['Test']):
        pckl_file = '{0}{1}.pckl'.format(corr_pckl, x).replace('XYZ',
                                                               method.lower())
        horiz_matrix.iloc[i, -1] = get_rmse(pckl_file, method)

    return horiz_matrix


def mean_var(mean_mat, method, corr_pckl, scint):
    # Create new column
    clmn = method + ' RMSE'
    mean_mat[clmn] = np.nan

    # Get mean vertical values
    for i,x in enumerate(mean_mat['Test']):
        pckl_file = '{0}{1}.pckl'.format(corr_pckl, x).replace('XYZ',
                                                               method.lower())
        mean_mat.iloc[i, -1] = get_rmse(pckl_file, method)

    # Calculate the mean values as needed
    pivot_mean = mean_mat.pivot_table(values=clmn, index=['KI %'],
                                      columns=['Nozzle Diameter (um)'],
                                      aggfunc=np.nanmean)

    # Create arrays from the pivot tables
    # Could plot directly from table but I didn't want to delve too deep
    RMSE_700 = pivot_mean[700]
    RMSE_700_fit = polyfit(KIconc, RMSE_700, 1)
    RMSE_700_r2 = 100 * RMSE_700_fit['determination']
    RMSE_700_label = 'y$_{{{0}}}$ = {1:.3f}x + {2:.3f}; R$^2$ {3:.0f}%' \
                     .format(700, RMSE_700_fit['polynomial'][0],
                             RMSE_700_fit['polynomial'][1], RMSE_700_r2)

    RMSE_2000 = pivot_mean[2000]
    RMSE_2000_fit = polyfit(KIconc, RMSE_2000, 1)
    RMSE_2000_r2 = 100 * RMSE_2000_fit['determination']
    RMSE_2000_label = 'y$_{{{0}}}$ = {1:.3f}x + {2:.3f}; R$^2$ {3:.0f}%' \
                      .format(2000, RMSE_2000_fit['polynomial'][0],
                              RMSE_2000_fit['polynomial'][1], RMSE_2000_r2)

    # RMSE plot (Peak = markers filled, Ellipse = markers empty)
    if method == 'Peak':
        fstyle = 'full'
    else:
        fstyle = 'none'

    plt.figure()
    plt.plot(KIconc, RMSE_700, fillstyle=fstyle, color='olivedrab',
             marker='s', label='700 um')
    plt.plot(KIconc, RMSE_700_fit['function'](KIconc), color='teal',
             label=RMSE_700_label)
    plt.plot(KIconc, RMSE_2000, fillstyle=fstyle, color='indianred',
             marker='^', label='2000 um')
    plt.plot(KIconc, RMSE_2000_fit['function'](KIconc), color='darkorange',
             label=RMSE_2000_label)
    plt.legend()
    plt.title('{0} - {1} RMSE'.format(scint, method))
    plt.xlabel('KI (%)')
    plt.ylabel('RMSE ($\mu$m)')
    plt.savefig('{0}/{1}_mean_{2}_.png'.format(plt_fld, scint, method))
    plt.close()

    return mean_mat, RMSE_700, RMSE_2000


def combi_var(mean_RMSE_700, mean_RMSE_2000, method):
    RMSE_combi = np.mean([mean_RMSE_700, mean_RMSE_2000], axis=0)
    RMSE_combi_fit = polyfit(KIconc, RMSE_combi, 1)
    RMSE_combi_r2 = 100 * RMSE_combi_fit['determination']
    RMSE_combi_label = 'y = {0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%' \
                       .format(RMSE_combi_fit['polynomial'][0],
                               RMSE_combi_fit['polynomial'][1], RMSE_combi_r2)


    return RMSE_combi, RMSE_combi_fit['function'], RMSE_combi_label


def main(scint, test_matrix):
    # Processed data sets location
    corr_pckl = '{0}/Corrected/{1}_XYZT/Summary/{1}_'.format(prj_fld, scint)

    ##########################################################################

    # Vertical variation (absolute error vs. vertical location plots)
    vert_matrix = test_matrix[~test_matrix['Test'].str.contains('mm')].copy()
    vert_matrix = vert_var(vert_matrix, 'Peak', corr_pckl, scint)
    vert_matrix = vert_var(vert_matrix, 'Ellipse', corr_pckl, scint)

    ##########################################################################

    # Horizontal variation (RMSE vs. horizontal location plots)
    horiz_matrix = test_matrix[test_matrix['Test'].str.contains('mm')].copy()

    # Create X position column
    x = 'X Position'
    horiz_matrix[x] = np.nan

    for i,n in enumerate(horiz_matrix['Test']):
        pckl_file = '{0}{1}.pckl'.format(corr_pckl, n).replace('XYZ', 'peak')
        horiz_matrix.iloc[i, -1] = get_xpos(pckl_file)

    horiz_matrix = horiz_var(horiz_matrix, 'Peak', corr_pckl)
    horiz_matrix = horiz_var(horiz_matrix, 'Ellipse', corr_pckl)

    # Horizontal plot
    plt.figure()
    plt.plot(horiz_matrix[x], horiz_matrix['Peak RMSE'],
             color='olivedrab', marker='s', label='Peak RMSE')
    plt.plot(horiz_matrix[x], horiz_matrix['Ellipse RMSE'], fillstyle='none',
             color='olivedrab', marker='s', label='Ellipse RMSE')
    plt.legend()
    plt.ylabel('RMSE ($\mu$m)')
    plt.xlabel('X Position (px)')
    plt.title('{0} - Horizontal Variation - 700 um, 10% KI'.format(scint))
    plt.savefig('{0}/{1}_horiz.png'.format(plt_fld, scint))
    plt.close()

    ##########################################################################

    # Mean vertical variation (RMSE vs. KI % plots for 700 & 2000 um)
    mean_mat = test_matrix[~test_matrix['Test'].str.contains('mm')].copy()

    # Get mean vertical values
    mean_mat, peakRMSE_700, peakRMSE_2000 = mean_var(mean_mat, 'Peak',
                                                     corr_pckl, scint)
    mean_mat, elpsRMSE_700, elpsRMSE_2000 = mean_var(mean_mat, 'Ellipse',
                                                     corr_pckl, scint)

    ##########################################################################

    # Combined mean variations (RMSE vs. KI % plots, diameters combined)
    peakRMSE = combi_var(peakRMSE_700, peakRMSE_2000, 'Peak')
    elpsRMSE = combi_var(elpsRMSE_700, elpsRMSE_2000, 'Ellipse')

    # Save the linear fitted correction factors
    cf_file = '{0}/Corrected/{1}'.format(prj_fld, scint)

    with open('{0}_{1}RMSE.txt'.format(cf_file, 'peak'), 'wb') as f:
        np.savetxt(f, np.c_[KIconc, peakRMSE[1](KIconc)])

    with open('{0}_{1}RMSE.txt'.format(cf_file, 'ellipse'), 'wb') as f:
        np.savetxt(f, np.c_[KIconc, elpsRMSE[1](KIconc)])

    plt.figure()
    plt.plot(KIconc, peakRMSE[0], color='lightcoral', marker='s',
             linestyle='', label='Peak RMSE', zorder=2)
    plt.plot(KIconc, peakRMSE[1](KIconc), linestyle='-', color='maroon',
             label=peakRMSE[2], zorder=1)
    plt.plot(KIconc, elpsRMSE[0], color='cornflowerblue', marker='^',
             linestyle='', label='Ellipse RMSE', zorder=2)
    plt.plot(KIconc, elpsRMSE[1](KIconc), linestyle='-', color='mediumblue',
             label=elpsRMSE[2], zorder=1)
    plt.title('{0} - Combined'.format(scint))
    plt.legend()
    plt.xlabel('KI (%)')
    plt.ylabel('RMSE ($\mu$m)')
    plt.savefig('{0}/{1}_combined.png'.format(plt_fld, scint))
    plt.close()


def run_main():
    # Scintillator
    scintillators = ['LuAG', 'YAG']

    # Test matrix
    test_matrix = pd.read_csv('{0}/APS White Beam.txt'.format(prj_fld),
                              sep='\t+', engine='python')

    # Crop down the test matrix
    test_matrix = test_matrix[['Test', 'Nozzle Diameter (um)', 'KI %']].copy()

    for scint in scintillators:
        main(scint, test_matrix)


if __name__ == '__main__':
    run_main()
