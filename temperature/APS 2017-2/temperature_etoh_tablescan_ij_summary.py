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

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'
folder = project_folder + '/Processed/Ethanol'

# Create summary of Temperature data sets (to find most consistent profile)
flds = glob.glob(folder + '/IJ Ramping/Temperature/*/')

for fld in flds:
	temp = fld.rsplit('/')[-1]
	files = glob.glob(fld + '/Profiles/profile*')
	names = [x.rsplit('/')[-1].rsplit('_')[-1].rsplit('.')[0] for x in files]
	df = pd.DataFrame(columns=['Mean','StD','CV'], index=names)
	for name, file in zip(names, files):
		data = np.loadtxt(file)
		df.loc[name] = pd.Series({'Mean': round(np.mean(data), 3), 'StD': round(np.std(data), 3), 'CV': round(np.std(data) / np.mean(data), 3)})
	df.to_csv(fld + '/' + temp + 'summary.txt', sep='\t')