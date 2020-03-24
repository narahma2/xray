# -*- coding: utf-8 -*-
"""
Compares the interpolated IJ Ramping temperatures from the calibration to the Nozzle T.
Constructs a thermometer using the IJ Ramping/Positions data sets.

@Author: narahma2
@Date:   2020-03-23 13:06:21
@Last Modified by:   narahma2
@Last Modified time: 2020-03-23 13:06:21
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

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Statistics.calc_statistics import polyfit

## Setup initial parameters
project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'
folder = project_folder + '/Processed/Ethanol'

# Select profiles to be used as thermometers
profiles = ['aratio', 'peakq', 'var', 'skew', 'kurt']

# Select calibration data sets
calibrations = ['408', '409']

# Load in IJ Ramping folders
flds = glob.glob(folder + '/IJ Ramping/Positions/y*/')

## Load in and process data sets
# Iterate through each selected calibration jet
for calibration in calibrations:
	# Initialize summary array
	summary = np.zeros((len(flds), len(profiles)))

	# Iterate through each selected profile
	for j, profile in enumerate(profiles):
		# Create polynomial object based on selected profile & calibration jet
		p = np.poly1d(np.loadtxt(folder + '/' + calibration + '/Statistics/' + profile + '_polynomial.txt'))
		
		# Create folders
		plots_folder = folder + '/IJ Ramping/PositionsInterp/' + calibration + '_' + profile
		if not os.path.exists(plots_folder):
			os.makedirs(plots_folder)

		# Load in Positions
		for i, fld in enumerate(flds):
			# Y position string (y00p25, y00p05, etc.)
			yp = fld.rsplit('/')[-2]

			# Profile data for the IJ Ramping Positions
			data = np.loadtxt(fld + '/Profiles/profile_' + profile + '.txt')

			# Nozzle T
			nozzleT = np.loadtxt(fld + '/temperature.txt')
			# Interpolated temperature
			interpT = p(data)
			# Fit a linear line of interpT vs. nozzleT
			fit = polyfit(interpT, nozzleT, 1)

			# Calculate RMSE
			rmse = np.sqrt(((interpT - nozzleT)**2).mean())

			# Build up summary
			summary[i,j] = rmse

			# Plot results
			plt.figure()
			plt.plot(interpT, nozzleT, ' o', markerfacecolor='none', markeredgecolor='b', label='Data')
			plt.plot(interpT, fit['function'](interpT), 'k', linewidth=2.0, label='y = ' + '%0.2f'%fit['polynomial'][0] + 'x + ' + '%0.2f'%fit['polynomial'][1])
			plt.title('y = ' + yp[1:].replace('p', '.') + ' mm - ' + calibration + ': ' + profile)
			plt.legend()
			plt.xlabel('Interpolated Temperature (°C)')
			plt.ylabel('Nozzle Temperature (°C)')
			plt.tight_layout()
			plt.savefig(plots_folder + '/' + yp + '.png')
			plt.close()

	# Save summary file
	np.savetxt(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_rmse.txt', summary, delimiter='\t', header=profiles)
