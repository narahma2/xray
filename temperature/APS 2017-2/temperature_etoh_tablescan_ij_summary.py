# -*- coding: utf-8 -*-
"""
Summarizes the processed IJ Ramping data sets.
See "X-ray Temperature/APS 2017-2/IJ Ethanol Ramping" in OneNote.

Created on Wedn March  18 11:55:00 2020

@author: rahmann
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
import numpy as np
import pandas as pd
from Statistics.calc_statistics import polyfit

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'
folder = project_folder + '/Processed/Ethanol'

## Create summary of Temperature data sets (to find most consistent profile)
flds = glob.glob(folder + '/IJ Ramping/Temperature/*/')

for fld in flds:
	temp = fld.rsplit('/')[-1]
	files = glob.glob(fld + '/Profiles/profile*')
	names = [x.rsplit('/')[-1].rsplit('_')[-1].rsplit('.')[0] for x in files]
	df = pd.DataFrame(columns=['R^2', 'Mean', 'StD', 'CfVar', 'CfDisp'], index=names)
	for name, file in zip(names, files):
		# Profile data
		data = np.loadtxt(file)

		# Y positions
		y = np.loadtxt(fld + '/positions.txt')

		# Calculate R^2
		r2 = polyfit(data, y, 1)['determination']

		# Calculate Coefficient of Variation
		mean = np.mean(data)
		std = np.std(data)
		cv = np.std(data) / np.mean(data)

		# Calculate Coefficient of Dispersion
		Q1 = np.percentile(data, 25, interpolation = 'midpoint')
		Q3 = np.percentile(data, 75, interpolation = 'midpoint')
		cd = (Q1 - Q3) / (Q1 + Q3)

		# Fill in specific profile in DataFrame
		df.loc[name] = pd.Series({'R^2': round(r2, 3), 'Mean': round(mean, 3), 'StD': round(std, 3), 'CfVar': round(cv, 3), 'CfDisp': round(cd, 3)})

	df.to_csv(fld + '/' + temp + 'summary.txt', sep='\t')

breakpoint()
## Summarize the Temperature summaries
flds = glob.glob(folder + '/IJ Ramping/Temperature/*/')
# Profile names (kurtosis, A1, q2, etc.)
profiles = glob.glob(flds[0] + '/Profiles/profile*')
names = [x.rsplit('/')[-1].rsplit('_')[-1].rsplit('.')[0] for x in profiles]
df = pd.DataFrame(columns=['Mean of R^2', 'StD of R^2', 'CfVar of R^2', 'CfDisp of R^2'], index=names)
r2_mean = [np.mean(pd.read_csv(fld + '/summary.txt', sep='\t', index_col=0, header=0).loc[name]['R^2']) for fld in flds for name in names]
	

## Create summary of Position data sets (to find best profile)
flds = glob.glob(folder + '/IJ Ramping/Positions/*/')

for fld in flds:
	pos = fld.rsplit('/')[-1]
	files = glob.glob(fld + '/Profiles/profile*')
	names = [x.rsplit('/')[-1].rsplit('_')[-1].rsplit('.')[0] for x in files]
	df = pd.DataFrame(columns=['R^2', 'Mean', 'StD', 'CfVar', 'CfDisp'], index=names)
	for name, file in zip(names, files):
		# Profile data
		data = np.loadtxt(file)

		# Temperature
		T = np.loadtxt(fld + '/temperature.txt')

		# Calculate R^2
		r2 = polyfit(data, T, 1)['determination']

		# Calculate Coefficient of Variation
		mean = np.mean(data)
		std = np.std(data)
		cv = np.std(data) / np.mean(data)

		# Calculate Coefficient of Dispersion
		Q1 = np.percentile(data, 25, interpolation = 'midpoint')
		Q3 = np.percentile(data, 75, interpolation = 'midpoint')
		cd = (Q1 - Q3) / (Q1 + Q3)

		# Fill in specific profile in DataFrame
		df.loc[name] = pd.Series({'R^2': round(r2, 3), 'Mean': round(mean, 3), 'StD': round(std, 3), 'CfVar': round(cv, 3), 'CfDisp': round(cd, 3)})

	df.to_csv(fld + '/' + pos + 'summary.txt', sep='\t')