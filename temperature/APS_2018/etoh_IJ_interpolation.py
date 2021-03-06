# @Author: naveed
# @Date:   2020-04-07 13:37:46
# @Last Modified by:   rahmann
# @Last Modified time: 2020-04-28 22:35:36

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from general.calc_statistics import polyfit

def main():
	# Setup initial parameters
	prj_fld = '/mnt/r/X-ray Temperature/APS 2018-1'
	folder = prj_fld + '/Processed/Ethanol'

	# Select profiles to be used as thermometers
	profiles = ['aratio', 'peak', 'peakq', 'var', 'skew', 'kurt', 'pca']

	# Select calibration data sets
	calibration = 'Combined'

	# Load in IJ Ramping folders
	flds = ['/IJ65C/Perpendicular', '/IJ65C/Transverse']

	## Load in and process data sets
	# Iterate through each selected profile
	for j, profile in enumerate(profiles):
		# Create polynomial object based on selected profile & calibration jet
		# Load APS 2018-1 for 'Combined'
		p = np.poly1d(np.loadtxt(folder.replace('APS 2017-2', 'APS 2018-1').replace('Ethanol', 'Ethanol_700umNozzle') + '/' + calibration + '/Statistics/' + profile + '_polynomial.txt'))

		# Load in Positions
		T = []
		for i, fld in enumerate(flds):
			# Create folders
			plots_folder = folder + fld + 'Interp'
			if not os.path.exists(plots_folder):
				os.makedirs(plots_folder)

			# Positions array
			positions = np.loadtxt(folder + fld + '/positions.txt')

			# Profile data for the IJ Ramping Positions
			data = np.loadtxt(folder + fld + '/Profiles/profile_' + profile + '.txt')

			# Nozzle T (65 deg C)
			nozzleT = 338
			# Interpolated temperature
			interpT = p(data)

			# Plot results
			plt.figure()
			plt.plot(positions, interpT, ' o', markerfacecolor='none', markeredgecolor='b', label='Data')
			plt.title('T = 338 K - ' + calibration + ': ' + profile)
			plt.legend()
			plt.xlabel('Y Location (mm)')
			plt.ylabel('Interpolated Temperature (K)')
			plt.tight_layout()
			plt.ylim([270, 350])
			plt.savefig(plots_folder + '/' + profile + '.png')
			plt.close()


if __name__ == '__main__':
	main()
