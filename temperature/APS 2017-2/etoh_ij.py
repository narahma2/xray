# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:29:08 2019

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
from scipy.constants import convert_temperature
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from calc_statistics import comparefit
from temperature_processing import main as temperature_processing

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'

tests = ['Ethanol/IJ Ambient']

for test in tests:
    folder = project_folder + '/Processed/Ethanol'

    if 'Ambient' in test:
        scans = [438, 439, 440, 441, 442, 443, 444]
        bg_scan = 437
        y = [0.1, 1, 2, 2.5, 4, 15, 25]
    if 'Hot' in test:
        scans = [428, 430, 431]
        bg_scan = 429
        y = [10, np.nan, 25]

    g = h5py.File(project_folder + '/RawData/Scan_' + str(bg_scan) + '.hdf5', 'r')
    bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
    bg_avg = np.mean(bg, axis=0)

    intensities = []
    temperatures = []
    for scan in scans:
        f = h5py.File(project_folder + '/RawData/Scan_' + str(scan) + '.hdf5', 'r')
        q = list(f['q'])
        intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        intensities.append(np.mean(intensity, axis=0))
        
        # Load temperature and convert to Kelvin
        temperature = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        temperatures.append(np.mean(temperature, axis=0))
        temperature = convert_temperature(temperature, 'Celsius', 'Kelvin')
    
    sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())
    intensities = [(x-bg_avg) for x in intensities]
    filtered_intensity = [savgol_filter(x, 55, 3) for x in intensities]
    reduced_q = np.array(q[sl])
    reduced_intensity = [x[sl] for x in filtered_intensity]
    reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]

    temperature_processing(test.rsplit('/')[0], folder, test.rsplit('/')[1], reduced_intensity, reduced_q, temperature=[], structure_factor=None, y=y, ramping=False)
