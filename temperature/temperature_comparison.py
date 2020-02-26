# -*- coding: utf-8 -*-
"""
Compares the temperature data sets.

Created on Thu Jan 16 11:18:19 2020

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
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats
from Form_Factor.xray_factor import ItoS, waas_kirf
from calc_statistics import polyfit

folder_2017 = sys_folder + '/X-ray Temperature/APS 2017-2/Processed'
#scan_2017 = '/Water/400'
scan_2017 = '/Ethanol/408'
folder_2018 = sys_folder + '/X-ray Temperature/APS 2018-1/Processed'
#scan_2018 = '/Water_700umNozzle/Combined'
scan_2018 = '/Ethanol_700umNozzle/RampDown'

with open(glob.glob(folder_2017 + scan_2017 + '/*.pckl')[0], 'rb') as f:
    temperature_2017, q_2017, intensity_2017 = pickle.load(f)
    
with open(glob.glob(folder_2018 + scan_2018 + '/*.pckl')[0], 'rb') as f:
    temperature_2018, q_2018, intensity_2018 = pickle.load(f)

stat = 'skew'
profile = np.loadtxt(folder_2017 + scan_2017 + '/Profiles/profile_' + stat + '.txt')        
calib = np.poly1d(np.loadtxt(folder_2018 + scan_2018 + '/Statistics/' + stat + '_polynomial.txt'))

plt.figure()
plt.plot(profile, temperature_2017, ' o', markerfacecolor='none', markeredgecolor='b', label='2017 Data')
plt.plot(profile, calib(profile), 'k', linewidth=2.0, label='2018 Calibration')
plt.xlabel(stat)
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()

fig, ax = plt.subplots()
plt.plot(temperature_2017, calib(profile), ' o', markerfacecolor='none', markeredgecolor='b', label='2017 Data')
ax.plot([0,1],[0,1], transform=ax.transAxes, color='k', linewidth=2.0)
plt.xlabel('Temperature 2017 (°C)')
plt.ylabel('Temperature 2018 (°C)')
plt.xlim([min(temperature_2017), max(temperature_2017)])
plt.ylim([min(temperature_2017), max(temperature_2017)])
plt.tight_layout()






#%%
#ij_scans = [452, 453, 448, 446, 447, 454, 455, 456]
#ij_scan = ij_scans[4]
#with open(project_folder + '/Water IJ Mixing/Scan' + str(ij_scan) + '/' + str(ij_scan) + '_data.pckl', 'rb') as f:
#    x_loc, reduced_q, reduced_intensity, q1, q2, q1_int, q2_int = pickle.load(f)
#
#plt.figure()
#plt.plot(x_loc, [q1[i]-reduced_q[266] for i,x in enumerate(reduced_intensity)])









