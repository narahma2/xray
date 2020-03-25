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
calibrations = ['408', '409', 'Combined']

# Load in IJ Ramping folders
flds = glob.glob(folder + '/IJ Ramping/Positions/y*/')

# Positions array
positions = np.loadtxt(folder + '/IJ Ramping/Temperature/T281p13/positions.txt')

## Load in and process data sets
# Iterate through each selected calibration jet
for calibration in calibrations:
	# Change path to APS 2018-1 for 'Combined'
	if calibration == 'Combined':
		folder = folder.replace('APS 2017-2', 'APS 2018-1').replace('Ethanol', 'Ethanol_700umNozzle')
			
	# Initialize summary arrays
	summary_rmse = np.zeros((len(flds), len(profiles)))
	summary_mape = np.zeros((len(flds), len(profiles)))
	summary_zeta = np.zeros((len(flds), len(profiles)))
	summary_mdlq = np.zeros((len(flds), len(profiles)))

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

			# Calculate RMSE (root mean squared error)
			rmse = np.sqrt(((interpT - nozzleT)**2).mean())

			# Calculate MAPE (mean absolute percentage error)
			mape = 100*np.sum(np.abs((interpT - nozzleT)/nozzleT)) / len(nozzleT)

			# Calculate median symmetric accuracy
			zeta = 100*np.exp(np.median(np.abs(np.log(interpT/nozzleT))) - 1)

			# Calculate MdLQ (median log accuracy ratio)
			# Positive/negative: systematic (over/under)-prediction
			mdlq = np.median(np.log(interpT/nozzleT))

			# Build up summaries
			summary_rmse[i,j] = rmse
			summary_mape[i,j] = mape
			summary_zeta[i,j] = zeta
			summary_mdlq[i,j] = mdlq

			# Plot results
			plt.figure()
			plt.plot(interpT, nozzleT, ' o', markerfacecolor='none', markeredgecolor='b', label='Data')
			plt.plot(nozzleT, nozzleT, 'k', linewidth=2.0, label='y = x')
			plt.plot(interpT, fit['function'](interpT), 'r', linewidth=2.0, label='y = ' + '%0.2f'%fit['polynomial'][0] + 'x + ' + '%0.2f'%fit['polynomial'][1])
			plt.title('y = ' + yp[1:].replace('p', '.') + ' mm - ' + calibration + ': ' + profile)
			plt.legend()
			plt.xlabel('Interpolated Temperature (K)')
			plt.ylabel('Nozzle Temperature (K)')
			plt.tight_layout()
			plt.savefig(plots_folder + '/' + yp + '.png')
			plt.close()

	# Plot summaries
	plt.figure()
	[plt.plot(positions, summary_rmse[:,j], linewidth=2.0, label=profiles[j]) for j in range(len(profiles))]
	plt.legend()
	plt.ylabel('RMSE (K)')
	plt.xlabel('Vertical Location (mm)')
	plt.title(calibration)
	plt.savefig(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_rmse.png')
	plt.close()

	plt.figure()
	[plt.plot(positions, summary_mape[:,j], linewidth=2.0, label=profiles[j]) for j in range(len(profiles))]
	plt.legend()
	plt.ylabel('MAPE (%)')
	plt.xlabel('Vertical Location (mm)')
	plt.title(calibration)
	plt.savefig(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_mape.png')
	plt.close()

	plt.figure()
	[plt.plot(positions, summary_zeta[:,j], linewidth=2.0, label=profiles[j]) for j in range(len(profiles))]
	plt.legend()
	plt.ylabel('$\zeta$ (%)')
	plt.xlabel('Vertical Location (mm)')
	plt.title(calibration)
	plt.savefig(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_zeta.png')
	plt.close()

	plt.figure()
	[plt.plot(positions, summary_mdlq[:,j], linewidth=2.0, label=profiles[j]) for j in range(len(profiles))]
	plt.legend()
	plt.ylabel('MdLQ (-)')
	plt.xlabel('Vertical Location (mm)')
	plt.title(calibration)
	plt.savefig(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_mdlq.png')
	plt.close()

	# Save summary file
	np.savetxt(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_rmse.txt', summary_rmse, delimiter='\t', header="\t".join(str(x) for x in profiles))
	np.savetxt(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_mape.txt', summary_mape, delimiter='\t', header="\t".join(str(x) for x in profiles))
	np.savetxt(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_zeta.txt', summary_zeta, delimiter='\t', header="\t".join(str(x) for x in profiles))
	np.savetxt(folder + '/IJ Ramping/PositionsInterp/' + calibration + '_mdlq.txt', summary_mdlq, delimiter='\t', header="\t".join(str(x) for x in profiles))