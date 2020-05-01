"""
-*- coding: utf-8 -*-
Summarize the correction factors for the jets.

@Author: rahmann
@Date:   2020-04-30 10:54:07
@Last Modified by:   rahmann
@Last Modified time: 2020-04-30 10:54:07
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

def get_mean_ellipseT(path):
	with open(path, 'rb') as f:
		processed_data = pickle.load(f)

	mean_ellipseT = np.nanmean(processed_data['Transmission Ratios'][0])

	return mean_ellipseT

def get_ellipseT(path):
	with open(path, 'rb') as f:
		processed_data = pickle.load(f)

	ellipseT = np.array(processed_data['Transmission Ratios'][0])

	return ellipseT


def get_mean_peakT(path):
	with open(path, 'rb') as f:
		processed_data = pickle.load(f)

	mean_peakT = np.nanmean(processed_data['Transmission Ratios'][1])

	return mean_peakT

def get_peakT(path):
	with open(path, 'rb') as f:
		processed_data = pickle.load(f)

	peakT = np.array(processed_data['Transmission Ratios'][1])

	return peakT


def main():
	# Location of APS 2018-1 data
	project_folder = '{0}/X-ray Radiography/APS 2018-1/'.format(sys_folder) 

	# Save location for the plots
	plots_folder = create_folder('{0}/Figures/Jet_Summary/'.format(project_folder))

	# Scintillator
	scintillators = ['LuAG', 'YAG']

	# KI %
	KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]

	# Test matrix
	test_matrix = pd.read_csv('{0}/APS White Beam.txt'.format(project_folder), sep='\t+', engine='python')

	# Crop down the test matrix
	test_matrix = test_matrix[['Test', 'Nozzle Diameter (um)', 'KI %']].copy()

	for scintillator in scintillators:
		# Processed data sets location
		processed_folder = '{0}/Processed/{1}/Summary/'.format(project_folder, scintillator)

		# Vertical variation
		vertical_matrix = test_matrix[~test_matrix['Test'].str.contains('mm')].copy()

		# Get vertical values
		vertical_matrix['Ratio Ellipse T'] = [get_ellipseT('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x)) for x in vertical_matrix['Test']]
		vertical_matrix['Ratio Peak T'] = [get_peakT('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x)) for x in vertical_matrix['Test']]
		vert_peakT_grouped = vertical_matrix.groupby(['Nozzle Diameter (um)', 'KI %']).apply(lambda x: np.mean(x['Ratio Peak T'], axis=0))
		vert_ellipseT_grouped = vertical_matrix.groupby(['Nozzle Diameter (um)', 'KI %']).apply(lambda x: np.mean(x['Ratio Ellipse T'], axis=0))
		axial_positions = np.linspace(start=20, stop=325, num=325-20+1, dtype=int)

		linecolors = ['dimgray', 'firebrick', 'goldenrod', 'mediumseagreen', 'steelblue', 'mediumpurple', 'hotpink']
		linelabels = ['0%', '1.6%', '3.4%', '4.8%', '8%', '10%', '11.1%']

		# Vertical peakT plot
		fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
		fig.set_size_inches(12, 6)
		[ax1.plot(axial_positions[2:-1], vert_peakT_grouped[700,KI_conc[i]][2:-1], color=x, linewidth=2.0) for i,x in enumerate(linecolors)]
		[ax2.plot(axial_positions[2:-1], vert_peakT_grouped[2000,KI_conc[i]][2:-1], color=x, linewidth=2.0) for i,x in enumerate(linecolors)]
		ax1.title.set_text(r'700 $\mu$m')
		ax2.title.set_text(r'2000 $\mu$m')
		ax1.set_xlabel('Axial Position (px)')
		ax2.set_xlabel('Axial Position (px)')
		ax1.set_ylabel('Correction Factor (-)')
		fig.suptitle('Vertical Variation - Ratio Peak T')
		fig.legend(ax1, labels=linelabels, loc="center right", borderaxespad=0.1, title='KI %')
		plt.subplots_adjust(wspace = 0.05, top = 0.90)
		plt.savefig('{0}/{1}_vert_peakT.png'.format(plots_folder, scintillator))
		plt.close()

		# Vertical ellipseT plot
		fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
		fig.set_size_inches(12, 6)
		[ax1.plot(axial_positions[2:-1], vert_ellipseT_grouped[700,KI_conc[i]][2:-1], linewidth=2.0) for i,x in enumerate(linecolors)]
		[ax2.plot(axial_positions[2:-1], vert_ellipseT_grouped[2000,KI_conc[i]][2:-1], linewidth=2.0) for i,x in enumerate(linecolors)]
		ax1.title.set_text(r'700 $\mu$m')
		ax2.title.set_text(r'2000 $\mu$m')
		ax1.set_xlabel('Axial Position (px)')
		ax2.set_xlabel('Axial Position (px)')
		ax1.set_ylabel('Correction Factor (-)')
		fig.suptitle('Vertical Variation - Ratio Ellipse T')
		fig.legend(ax1, labels=linelabels, loc="center right", borderaxespad=0.1, title='KI %')
		plt.subplots_adjust(wspace = 0.05, top = 0.90)
		plt.savefig('{0}/{1}_vert_ellipseT.png'.format(plots_folder, scintillator))

		#############################################################################

		# Horizontal variation
		horiz_matrix = test_matrix[test_matrix['Test'].str.contains('mm')].copy()
		horiz_matrix['X Position'] = [get_xpos('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x)) for x in horiz_matrix['Test']]

		# Sort horiz_matrix by X position, reset index, and drop the edge outliers
		horiz_matrix.sort_values(by=['X Position'], inplace=True)
		horiz_matrix.reset_index(inplace=True)
		horiz_matrix.drop([0, len(horiz_matrix)-1], inplace=True)

		# Get horizontal values
		horiz_matrix['Ratio Ellipse T'] = [get_mean_ellipseT('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x)) for x in horiz_matrix['Test']]
		horiz_matrix['Ratio Peak T'] = [get_mean_peakT('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x)) for x in horiz_matrix['Test']]

		# Horizontal plot
		plt.figure()
		plt.plot(horiz_matrix['X Position'], horiz_matrix['Ratio Peak T'], color='olivedrab', marker='s', label='Ratio Peak T')
		plt.plot(horiz_matrix['X Position'], horiz_matrix['Ratio Ellipse T'], fillstyle='none', color='olivedrab', marker='s', label='Ratio Ellipse T')
		plt.legend()
		plt.title('{0} - Horizontal Variation - 700 um, 10% KI'.format(scintillator))
		plt.savefig('{0}/{1}_horiz.png'.format(plots_folder, scintillator))
		plt.close()

		#############################################################################

		# Mean vertical variation
		mean_matrix = test_matrix[~test_matrix['Test'].str.contains('mm')].copy()

		# Get mean vertical values
		mean_matrix['Ratio Ellipse T'] = [get_mean_ellipseT('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x)) for x in mean_matrix['Test']]
		mean_matrix['Ratio Peak T'] = [get_mean_peakT('{0}/{1}_{2}.pckl'.format(processed_folder, scintillator, x)) for x in mean_matrix['Test']]
		
		# Calculate the mean values as needed
		pivot_mean_peakT = mean_matrix.pivot_table(values='Ratio Peak T', index=['KI %'], columns=['Nozzle Diameter (um)'], aggfunc=np.nanmean)
		pivot_mean_ellipseT = mean_matrix.pivot_table(values='Ratio Ellipse T', index=['KI %'], columns=['Nozzle Diameter (um)'], aggfunc=np.nanmean)
				# Create arrays from the pivot tables 
		# Could plot directly from the pivot table but I didn't want to delve too deep into that
		mean_peakT_700 = pivot_mean_peakT[700]
		peakT_700_fit = polyfit(KI_conc, mean_peakT_700, 1)
		peakT_700_fit_r2 = peakT_700_fit['determination']
		peakT_700_fit_label = 'y$_{700}$ = ' + '{0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'.format(peakT_700_fit['polynomial'][0], peakT_700_fit['polynomial'][1], 100*peakT_700_fit_r2)
		
		mean_peakT_2000 = pivot_mean_peakT[2000]
		peakT_2000_fit = polyfit(KI_conc, mean_peakT_2000, 1)
		peakT_2000_fit_r2 = peakT_2000_fit['determination']
		peakT_2000_fit_label = 'y$_{2000}$ = ' + '{0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'.format(peakT_2000_fit['polynomial'][0], peakT_2000_fit['polynomial'][1], 100*peakT_2000_fit_r2)
		
		mean_ellipseT_700 = pivot_mean_ellipseT[700]
		ellipseT_700_fit = polyfit(KI_conc, mean_ellipseT_700, 1)
		ellipseT_700_fit_r2 = ellipseT_700_fit['determination']
		ellipseT_700_fit_label = 'y$_{700}$ = ' + '{0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'.format(ellipseT_700_fit['polynomial'][0], ellipseT_700_fit['polynomial'][1], 100*ellipseT_700_fit_r2)

		mean_ellipseT_2000 = pivot_mean_ellipseT[2000]
		ellipseT_2000_fit = polyfit(KI_conc, mean_ellipseT_2000, 1)
		ellipseT_2000_fit_r2 = ellipseT_2000_fit['determination']
		ellipseT_2000_fit_label = 'y$_{2000}$ = ' + '{0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'.format(ellipseT_2000_fit['polynomial'][0], ellipseT_2000_fit['polynomial'][1], 100*ellipseT_2000_fit_r2)

		# PeakT plot (markers filled)
		plt.figure()
		plt.plot(KI_conc, mean_peakT_700, color='olivedrab', marker='s', label='700 um')
		plt.plot(KI_conc, peakT_700_fit['function'](KI_conc), color='teal', label=peakT_700_fit_label)
		plt.plot(KI_conc, mean_peakT_2000, color='indianred', marker='^', label='2000 um')
		plt.plot(KI_conc, peakT_2000_fit['function'](KI_conc), color='darkorange', label=peakT_2000_fit_label)
		plt.legend()
		plt.title('{0} - Ratio Peak T'.format(scintillator))
		plt.xlabel('KI (%)')
		plt.ylabel('Correction Factor (-)')
		plt.savefig('{0}/{1}_mean_peakT.png'.format(plots_folder, scintillator))
		plt.close()

		# EllipseT plot (markers not filled)
		plt.figure()
		plt.plot(KI_conc, mean_ellipseT_700, fillstyle='none', color='olivedrab', marker='s', label='700 um')
		plt.plot(KI_conc, ellipseT_700_fit['function'](KI_conc), color='teal', label=ellipseT_700_fit_label)
		plt.plot(KI_conc, mean_ellipseT_2000, fillstyle='none', color='indianred', marker='^', label='2000 um')
		plt.plot(KI_conc, ellipseT_2000_fit['function'](KI_conc), color='darkorange', label=ellipseT_2000_fit_label)
		plt.legend()
		plt.title('{0} - Ratio Ellipse T'.format(scintillator))
		plt.xlabel('KI (%)')
		plt.ylabel('Correction Factor (-)')
		plt.savefig('{0}/{1}_mean_ellipseT.png'.format(plots_folder, scintillator))
		plt.close()

		#############################################################################

		mean_peakT_combined = np.mean([mean_peakT_700, mean_peakT_2000], axis=0)
		peakT_combined_fit = polyfit(KI_conc, mean_peakT_combined, 1)
		peakT_combined_fit_r2 = peakT_combined_fit['determination']
		peakT_combined_fit_label = 'y = {0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'.format(peakT_combined_fit['polynomial'][0], peakT_combined_fit['polynomial'][1], 100*peakT_combined_fit_r2)

		mean_ellipseT_combined = np.mean([mean_ellipseT_700, mean_ellipseT_2000], axis=0)
		ellipseT_combined_fit = polyfit(KI_conc, mean_ellipseT_combined, 1)
		ellipseT_combined_fit_r2 = ellipseT_combined_fit['determination']
		ellipseT_combined_fit_label = 'y = {0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'.format(ellipseT_combined_fit['polynomial'][0], ellipseT_combined_fit['polynomial'][1], 100*ellipseT_combined_fit_r2)
		
		# Save the linear fitted correction factors
		with open('{0}/Processed/{1}/{1}_peakT_cf.txt'.format(project_folder, scintillator), 'wb') as f:
			np.savetxt(f, peakT_combined_fit['function'](KI_conc))

		with open('{0}/Processed/{1}/{1}_ellipseT_cf.txt'.format(project_folder, scintillator), 'wb') as f:
			np.savetxt(f, ellipseT_combined_fit['function'](KI_conc))

		plt.figure()
		plt.plot(KI_conc, mean_peakT_combined, color='lightcoral', marker='s', linestyle='', label='Ratio Peak T', zorder=2)
		plt.plot(KI_conc, peakT_combined_fit['function'](KI_conc), linestyle='-', color='maroon', label=peakT_combined_fit_label, zorder=1)
		plt.plot(KI_conc, mean_ellipseT_combined, color='cornflowerblue', marker='^', linestyle='', label='Ratio Ellipse T', zorder=2)
		plt.plot(KI_conc, ellipseT_combined_fit['function'](KI_conc), linestyle='-', color='mediumblue', label=ellipseT_combined_fit_label, zorder=1)
		plt.title('{0} - Combined'.format(scintillator))
		plt.legend()
		plt.xlabel('KI (%)')
		plt.ylabel('Correction Factor (-)')
		plt.savefig('{0}/{1}_combined.png'.format(plots_folder, scintillator))
		plt.close()


if __name__ == '__main__':
	main()