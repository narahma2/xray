"""
-*- coding: utf-8 -*-
Interpolation for the IJ Ambient data set. 
@Author: naveed
@Date:   2020-04-07 16:49:03
@Last Modified by:   naveed
@Last Modified time: 2020-04-07 16:54:03
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
profiles = ['aratio', 'peak', 'peakq', 'var', 'skew', 'kurt']

# Select calibration data sets
calibration = '409'

# Load in IJ Ramping folders
flds = ['/IJ Ambient']

## Load in and process data sets
# Iterate through each selected profile
for j, profile in enumerate(profiles):
	# Create polynomial object based on selected profile & calibration jet
	# Load APS 2018-1 for 'Combined'
	p = np.poly1d(np.loadtxt(folder + '/' + calibration + '/Statistics/' + profile + '_polynomial.txt'))

	# Load in Positions
	T = []
	for i, fld in enumerate(flds):
		# Create folders
		plots_folder = folder + fld + ' Interp'
		if not os.path.exists(plots_folder):
			os.makedirs(plots_folder)

		# Positions array
		positions = np.loadtxt(folder + fld + '/positions.txt')

		# Profile data for the IJ Ramping Positions
		data = np.loadtxt(folder + fld + '/Profiles/profile_' + profile + '.txt')

		# Nozzle T (65 deg C)
		nozzleT = 298
		# Interpolated temperature
		interpT = p(data)

		# Plot results
		plt.figure()
		plt.plot(positions, interpT, ' o', markerfacecolor='none', markeredgecolor='b', label='Data')
		plt.title('T = 298 K - ' + calibration + ': ' + profile)
		plt.legend()
		plt.xlabel('Y Location (mm)')
		plt.ylabel('Interpolated Temperature (K)')
		plt.tight_layout()
		plt.ylim([280, 350])
		plt.savefig(plots_folder + '/' + profile + '.png')
		plt.close()