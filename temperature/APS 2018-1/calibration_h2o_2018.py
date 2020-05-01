# -*- coding: utf-8 -*-
"""
Processes the APS 2018-1 temperature data sets.

Created on Thu Jan 16 17:08:33 2020

@author: rahmann
"""

import sys
if sys.platform == 'win32':
	sys.path.append('E:/GitHub/xray/general')
	sys.path.append('E:/GitHub/xray/temperature')
	sys_folder = 'R:/'
elif sys.platform == 'linux':
	sys.path.append('/mnt/e/GitHub/xray/general')
	sys.path.append('/mnt/e/GitHub/xray/temperature')
	sys_folder = '/mnt/r/'

import glob
import numpy as np
import os
from scipy.constants import convert_temperature
from scipy.signal import savgol_filter
from Form_Factor.xray_factor import ItoS
from temperature_processing import main as temperature_processing

def main():
	#%% Setup
	project_folder = sys_folder + '/X-ray Temperature/APS 2018-1/'

	test = 'Water_700umNozzle'

	folder = project_folder + 'Processed/' + test
	if not os.path.exists(folder):
		os.makedirs(folder)
		
	scans = ['/RampUp', '/RampDown', '/Combined']

	for scan in scans:
		#%% Load background
		bg = glob.glob(project_folder + '/Q Space/Water_700umNozzle/*bkgd*')
		q = np.loadtxt(bg[0], usecols=0)                    # Load q
		bg = [np.loadtxt(x, usecols=1) for x in bg]         # Load intensity
		
		# Average background intensities
		bg = np.mean(bg, axis=0)

		#%% Intensity correction
		rampUp = [1079, 1080, 1081, 1082, 1083, 1084, 1085, 1087, 1088]
		rampDown = [1089, 1090, 1092, 1093, 1095, 1096, 1097, 1098, 1099, 1100]
		combined = rampUp + rampDown
		
		if 'Up' in scan:
			files = rampUp
		elif 'Down' in scan:
			files = rampDown
		else:
			files = combined
			
		files = [glob.glob(project_folder + '/Q Space/Water_700umNozzle/*' + str(x) + '*') for x in files]

		# Set q range
		sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
		reduced_q = q[sl]

		# Load in temperatures and sort
		temperature = np.array([float(x[0].rsplit('Target')[-1].rsplit('_')[0].replace('p','.')) for x in files])

		# Convert temperature to Kelvin
		temperature = convert_temperature(temperature, 'Celsius', 'Kelvin')
		
		# Load in and process intensity curves (sort, filter, crop, normalize, ItoS)
		intensity = np.array([np.mean([np.loadtxt(x, usecols=1)-bg for x in y], axis=0) for y in files])
		filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]
		reduced_intensity = [x[sl] for x in filtered_intensity]
		reduced_intensity = np.array([y/np.trapz(y, x=reduced_q) for y in reduced_intensity])
		structure_factor = np.array([ItoS(np.array(reduced_q), x) for x in reduced_intensity])

		temperature_processing(test, folder, scan, reduced_intensity, reduced_q, temperature, structure_factor)        
		

if __name__ == '__main__':
	main()