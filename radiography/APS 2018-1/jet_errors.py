"""
-*- coding: utf-8 -*-
Summarize the correction factors for the jets.

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
from Statistics.CIs_LinearRegression import run_all, lin_fit, conf_calc, ylines_calc, plot_linreg_CIs
from Statistics.calc_statistics import polyfit
from jet_processing import create_folder
from timeit import default_timer as timer

def get_xpos(path):
	with open(path, 'rb') as f:
		processed_data = pickle.load(f)

	xpos = np.nanmean(processed_data['Lateral Position'])

	return xpos

def get_mean_rmse(path, method):
	with open(path, 'rb') as f:
		processed_data = pickle.load(f)

	# Return RMSE in mm
	mean_rmse = 10 * np.nanmean(processed_data[method + ' Errors'][0])

	return mean_rmse

def get_rmse(path, method):
	with open(path, 'rb') as f:
		processed_data = pickle.load(f)

	# Return RMSE in mm
	rmse = 10 * np.array(processed_data[method + ' Errors'][0])

	return rmse

def vertical_variation(vertical_matrix, method, plots_folder):
	# Get vertical values
	vertical_matrix[method + ' RMSE'] = [get_rmse('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x),
                                             method) for x in vertical_matrix['Test']]
	vert_RMSE_grouped = vertical_matrix.groupby(['Nozzle Diameter (um)', 'KI%'])
                                           .apply(lambda x: np.mean(x[method + ' RMSE'], axis=0))
	axial_positions = np.linspace(start=20, stop=325, num=325-20+1, dtype=int)

	linecolors = ['dimgray', 'firebrick', 'goldenrod', 'mediumseagreen', 'steelblue', 'mediumpurple', 'hotpink']
	linelabels = ['0%', '1.6%', '3.4%', '4.8%', '8%', '10%', '11.1%']

	warnings.filterwarnings('ignore')
	# Vertical RMSE plot
	fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	fig.set_size_inches(12, 6)

        # Plot the 700 um jets
        for i,x in enumerate(linecolors):
            ax1.plot(axial_positions[2:-1], vert_RMSE_grouped[700,KI_conc[i]][2:-1], color=x, linewidth=2.0)

        # Plot the 2000 um jets
        for i,x in enumerate(linecolors):
            ax2.plot(axial_positions[2:-1], vert_RMSE_grouped[2000,KI_conc[i]][2:-1], color=x, linewidth=2.0)

	ax1.title.set_text(r'700 $\mu$m')
	ax2.title.set_text(r'2000 $\mu$m')
	ax1.set_xlabel('Axial Position (px)')
	ax2.set_xlabel('Axial Position (px)')
	ax1.set_ylabel('RMSE (mm)')
	fig.suptitle('Vertical Variation - ' + method + ' RMSE')
	fig.legend(ax1, labels=linelabels, loc="center right", borderaxespad=0.1, title='KI %')
	plt.subplots_adjust(wspace = 0.05, top = 0.90)
	plt.savefig('{0}/{1}_vert_{2}RMSE.png'.format(plots_folder, scintillator, method))
	plt.close()
	warnings.filterwarnings('default')

	return vertical_matrix


def horizontal_variation(horiz_matrix, method):
	# Sort horiz_matrix by X position, reset index, and drop the edge outliers
	horiz_matrix.sort_values(by=['X Position'], inplace=True)
	horiz_matrix.reset_index(inplace=True)
	horiz_matrix.drop([0, len(horiz_matrix)-1], inplace=True)

	# Get horizontal values
	horiz_matrix[method + ' RMSE'] = [get_mean_rmse('{0}/{1}_{2}.pckl'.format(processed_folder,scintillator, x),
                                        method) for x in horiz_matrix['Test']]

        return horiz_matrix


def mean_variation(mean_matrix, method):
    # Get mean vertical values
    mean_matrix[method + ' RMSE'] = [get_mean_rmse('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x),
                                     method) for x in mean_matrix['Test']]

    # Calculate the mean values as needed
    pivot_mean = mean_matrix.pivot_table(values=method + ' RMSE', index=['KI %'],
                                         columns=['Nozzle Diameter (um)'], aggfunc=np.nanmean)

    # Create arrays from the pivot tables 
    # Could plot directly from the pivot table but I didn't want to delve too deep into that
    mean_RMSE_700 = pivot_mean[700]
    RMSE_700_fit = polyfit(KI_conc, mean_RMSE_700, 1)
    RMSE_700_fit_r2 = RMSE_700_fit['determination']
    RMSE_700_fit_label = 'y$_{700}$ = ' + '{0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'
                         .format(RMSE_700_fit['polynomial'][0], RMSE_700_fit['polynomial'][1], RMSE_700_fit_r2)

    mean_RMSE_2000 = pivot_mean[2000]
    RMSE_2000_fit = polyfit(KI_conc, mean_RMSE_2000, 1)
    RMSE_2000_fit_r2 = RMSE_2000_fit['determination']
    RMSE_2000_fit_label = 'y$_{2000}$ = ' + '{0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'
                          .format(RMSE_2000_fit['polynomial'][0], RMSE_2000_fit['polynomial'][1], RMSE_2000_fit_r2)

    # RMSE plot (Peak = markers filled, Ellipse = markers empty)
    if method == 'Peak':
        fstyle = 'full'
    else:
        fstyle = 'none'

    plt.figure()
    plt.plot(KI_conc, mean_RMSE_700, fillstyle=fstyle, color='olivedrab', marker='s', label='700 um')
    plt.plot(KI_conc, RMSE_700_fit['function'](KI_conc), color='teal', label=RMSE_700_fit_label)
    plt.plot(KI_conc, mean_RMSE_2000, fillstyle=fstyle, color='indianred', marker='^', label='2000 um')
    plt.plot(KI_conc, RMSE_2000_fit['function'](KI_conc), color='darkorange', label=RMSE_2000_fit_label)
    plt.legend()
    plt.title('{0} - {1} RMSE'.format(scintillator, method))
    plt.xlabel('KI (%)')
    plt.ylabel('RMSE (mm)')
    plt.savefig('{0}/{1}_mean_{2}_.png'.format(plots_folder, scintillator, method))
    plt.close()

    return mean_matrix, mean_RMSE_700, mean_RMSE_2000


def combined_variation(mean_RMSE_700, mean_RMSE_2000, method, project_folder, scintillator):
    mean_RMSE_combined = np.mean([mean_RMSE_700, mean_RMSE_2000], axis=0)
    RMSE_combined_fit = polyfit(KI_conc, mean_RMSE_combined, 1)
    RMSE_combined_fit_r2 = RMSE_combined_fit['determination']
    RMSE_combined_fit_label = 'y = {0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'
                              .format(RMSE_combined_fit['polynomial'][0],
                                      RMSE_combined_fit['polynomial'][1], RMSE_combined_fit_r2)

    # Save the linear fitted correction factors
    with open('{0}/Processed/{1}/{1}_{2}RMSE_cf.txt'.format(project_folder, scintillator, method), 'wb') as f:
            np.savetxt(f, RMSE_combined_fit['function'](KI_conc))

    return mean_RMSE_combined, RMSE_combined_fit, RMSE_combined_fit_label


def main(project_folder, plots_folder, scintillator, KI_conc, test_matrix)):
    # Processed data sets location
    processed_folder = '{0}/Processed/{1}/Summary/'.format(project_folder, scintillator)

    #############################################################################

    # Vertical variation
    vertical_matrix = test_matrix[~test_matrix['Test'].str.contains('mm')].copy()
    vertical_matrix = vertical_variation(vertical_matrix, 'Peak')
    vertical_matrix = vertical_variation(vertical_matrix, 'Ellipse')

    #############################################################################

    # Horizontal variation
    horiz_matrix = test_matrix[test_matrix['Test'].str.contains('mm')].copy()
    horiz_matrix['X Position'] = [get_xpos('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x))
                                  for x in horiz_matrix['Test']]
    horiz_matrix = horizontal_variation(horiz_matrix, 'Peak')
    horiz_matrix = horizontal_variation(horiz_matrix, 'Ellipse')

    # Horizontal plot
    plt.figure()
    plt.plot(horiz_matrix['X Position'], horiz_matrix['Peak RMSE'], color='olivedrab', marker='s', label='Peak RMSE')
    plt.plot(horiz_matrix['X Position'], horiz_matrix['Ellipse RMSE'], fillstyle='none', color='olivedrab',
             marker='s', label='Ellipse RMSE')
    plt.legend()
    plt.ylabel('RMSE (mm)')
    plt.xlabel('X Position (px)')
    plt.title('{0} - Horizontal Variation - 700 um, 10% KI'.format(scintillator))
    plt.savefig('{0}/{1}_horiz.png'.format(plots_folder, scintillator))
    plt.close()

    #############################################################################

    # Mean vertical variation
    mean_matrix = test_matrix[~test_matrix['Test'].str.contains('mm')].copy()

    # Get mean vertical values
    mean_matrix, mean_peakRMSE_700, mean_peakRMSE_2000 = mean_variation(mean_matrix, 'Peak')
    mean_matrix, mean_ellipseRMSE_700, mean_ellipseRMSE_2000 = mean_variation(mean_matrix, 'Ellipse')

    #############################################################################

    # Combined mean variations
    peakRMSE, peakRMSE_fit, peakRMSE_label = combined_variation(mean_peakRMSE_700, mean_peakRMSE_2000,
                                                                'Peak', project_folder, scintillator)

    ellipseRMSE, ellipseRMSE_fit, ellipseRMSE_label = combined_variation(mean_ellipseRMSE_700, mean_ellipseRMSE_2000,
                                                                         'Ellipse', project_folder, scintillator)
    plt.figure()
    plt.plot(KI_conc, peakRMSE, color='lightcoral', marker='s', linestyle='', label='Peak RMSE', zorder=2)
    plt.plot(KI_conc, peakRMSE_fit['function'](KI_conc), linestyle='-', color='maroon', label=peakRMSE_label, zorder=1)
    plt.plot(KI_conc, ellipseRMSE, color='cornflowerblue', marker='^', linestyle='', label='Ellipse RMSE', zorder=2)
    plt.plot(KI_conc, ellipseRMSE_fit['function'](KI_conc), linestyle='-', color='mediumblue',
             label=ellipseRMSE_label, zorder=1)
    plt.title('{0} - Combined'.format(scintillator))
    plt.legend()
    plt.xlabel('KI (%)')
    plt.ylabel('RMSE (mm)')
    plt.savefig('{0}/{1}_combined.png'.format(plots_folder, scintillator))
    plt.close()


def run_main():
    # Location of APS 2018-1 data
    project_folder = '{0}/X-ray Radiography/APS 2018-1/'.format(sys_folder)

    # Save location for the plots
    plots_folder = create_folder('{0}/Figures/Jet_Errors/'.format(project_folder))

    # Scintillator
    scintillators = ['LuAG', 'YAG']

    # KI %
    KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]

    # Test matrix
    test_matrix = pd.read_csv('{0}/APS White Beam.txt'.format(project_folder), sep='\t+', engine='python')

    # Crop down the test matrix
    test_matrix = test_matrix[['Test', 'Nozzle Diameter (um)', 'KI %']].copy()

    for scintillator in scintillators:
            main(project_folder, plots_folder, scintillator, KI_conc, test_matrix)


if __name__ == '__main__':
	run_main()
