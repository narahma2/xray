# -*- coding: utf-8 -*-
"""
Compares the interpolated IJ Ramping temperatures from the calibration to the Nozzle T.
This script tests how much the temperature changes at the different positions.

@Author: naveed
@Date:   2020-04-07 12:40:31
@Last Modified by:   naveed
@Last Modified time: 2020-04-07 12:37:33
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

def main():
	## Setup initial parameters
	project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'
	folder = project_folder + '/Processed/Ethanol'

	# Select profiles to be used as thermometers
	profiles = ['aratio', 'peak', 'peakq', 'var', 'skew', 'kurt', 'pca']

	# Select calibration data sets
	calibrations = ['408','409','Combined']

	# Load in IJ Ramping folders
	flds = glob.glob(folder + '/IJ Ramping/Temperature/T*/')

	# Positions array
	positions = np.loadtxt(glob.glob(folder + '/IJ Ramping/Temperature/T*')[0] + '/positions.txt')

	## Load in and process data sets
	# Iterate through each selected calibration jet
	for calibration in calibrations:
		# Iterate through each selected profile
		for j, profile in enumerate(profiles):
			# Create polynomial object based on selected profile & calibration jet
			# Load APS 2018-1 for 'Combined'
			if calibration == 'Combined':
				p = np.poly1d(np.loadtxt(folder.replace('APS 2017-2', 'APS 2018-1').replace('Ethanol', 'Ethanol_700umNozzle') + '/' + calibration + '/Statistics/' + profile + '_polynomial.txt'))
			else:
				p = np.poly1d(np.loadtxt(folder + '/' + calibration + '/Statistics/' + profile + '_polynomial.txt'))

			# Create folders
			plots_folder = folder + '/IJ Ramping/TemperatureInterp/' + calibration + '_' + profile
			if not os.path.exists(plots_folder):
				os.makedirs(plots_folder)

			# Load in Positions
			T = []
			combined_nozzleT = []
			combined_interpT = []
			for i, fld in enumerate(flds):
				# Y position string (y00p25, y00p05, etc.)
				Tp = fld.rsplit('/')[-2]
				T.append(float(Tp[1:].replace('p','.')))

				# Profile data for the IJ Ramping Positions
				data = np.loadtxt(fld + '/Profiles/profile_' + profile + '.txt')

				# Nozzle T
				nozzleT = T[-1]
				# Interpolated temperature
				interpT = p(data)

				combined_nozzleT.append(T[-1])
				combined_interpT.append(interpT)

				# Plot results
				plt.figure()
				plt.plot(positions, interpT, ' o', markerfacecolor='none', markeredgecolor='b', label='Data')
				plt.title('T = ' + Tp[1:].replace('p', '.') + ' K - ' + calibration + ': ' + profile)
				plt.legend()
				plt.xlabel('Y Location (mm)')
				plt.ylabel('Interpolated Temperature (K)')
				plt.tight_layout()
				plt.ylim([280, 350])
				plt.savefig(plots_folder + '/' + Tp + '.png')
				plt.close()

			slices = [2, 3, 6, 9, 10, 13]
			# Plot combined results
			plt.figure()
			[plt.plot(positions, combined_interpT[x], '-o', color='C'+str(i), label=str(round(combined_nozzleT[x])) + ' K') for i, x in enumerate(slices)]
			plt.title(calibration + ': ' + profile)
			plt.legend(title='Nozzle T', loc='upper right')
			plt.xlabel('Y Location (mm)')
			plt.ylabel('Interpolated Temperature (K)')
			plt.tight_layout()
			plt.ylim([260, 350])
			plt.savefig(plots_folder + '/combined.png')
			plt.close()


if __name__ == '__main__':
	main()