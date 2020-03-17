# -*- coding: utf-8 -*-
"""
Processes the ethanol impinging jet table scan (425).
See "X-ray Temperature/APS 2017-2/IJ Ethanol Ramping" in OneNote.

Created on Sat Apr  6 10:15:38 2019

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

import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from calc_statistics import comparefit
from temperature_processing import main as temperature_processing

test = 'Ethanol/IJ Ramping'
calib_no = [408]
rank_x = np.linspace(0,13, num=14, dtype=int)

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'
folder = project_folder + '/Processed/Ethanol'

f = h5py.File(project_folder + '/RawData/Scan_' + str(425) + '.hdf5', 'r')
g = h5py.File(project_folder + '/RawData/Scan_' + str(429) + '.hdf5', 'r')
bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
bg_avg = np.mean(bg, axis=0)

q = list(f['q'])
intensities = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]

sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())

nozzle_T = []
y_loc = []

for i in rank_x:
	y_locs = list(f['Rank_2_Point_' + str(i) + '/7bmb1:aero:m1.VAL'])
	for n, y in enumerate(y_locs):
		nozzle_T.append(list(f['Rank_2_Point_' + str(i) + '/7bm_dau1:dau:010:ADC'])[n])
		y_loc.append(round(list(f['Rank_2_Point_' + str(i) + '/7bmb1:aero:m1.VAL'])[n], 2))
		
# Convert temperatures to numpy array
nozzle_T = np.array(nozzle_T)

positions = []
for k in y_locs:
	positions.append([i for i, x in enumerate(y_loc) if x == round(k, 2)])      # 14 total occurrences of the 18 y positions        

# Read only the first 252 intensity scans (there's 277, seems to be extraneous data?)
intensities = intensities[0:252]

# Background subtraction
intensities = [(x-bg_avg) for x in intensities]

# Smooth out data using Savitzky-Golay filter
filtered_intensity = [savgol_filter(x, 55, 3) for x in intensities]

# Crop q range to look at same window as EtOH calibration sets
reduced_q = np.array(q[sl])
reduced_intensity = [x[sl] for x in filtered_intensity]

# Normalize intensity by the area (same process as the Nature paper)
reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]

# Normalize intensities by the curve's maximum
reduced_intensity = [z/np.max(z) for z in reduced_intensity]

# Convert reduced_intensity to numpy array
reduced_intensity = np.array(reduced_intensity)

maxVal = []
for i in range(len(positions)):
	maxVal.append(np.max([reduced_intensity[z] for z in positions[i]]))

for i in range(len(reduced_intensity)):
	find_yposition = (np.abs(y_loc[i] - np.array(y_locs))).argmin()
	reduced_intensity[i] = reduced_intensity[i] / maxVal[find_yposition]

for x, i in enumerate(positions[0]):
	y = y_loc[18*x:18*(x+1)]
	temperature = np.mean(nozzle_T[18*x:18*(x+1)])
	temperature_processing(test.rsplit('/')[0], folder, test.rsplit('/')[1] + '/Temperature/' + '{0:.2f}'.format(temperature).replace('.', 'p'), reduced_intensity[18*x:18*(x+1)], reduced_q, temperature=None, structure_factor=None, y=y, ramping=True)

for x, i in enumerate(positions):
	y = y_locs[x]
	temperature_processing(test.rsplit('/')[0], folder, test.rsplit('/')[1] + '/Positions/y' + '{0:.2f}'.format(y).replace('.', 'p'), reduced_intensity[i], reduced_q, temperature=nozzle_T[i], structure_factor=None, y=None, ramping=True)