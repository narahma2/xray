# -*- coding: utf-8 -*-
"""
Created on Sun March  1 13:22:00 2020

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
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from temperature_processing import main as temperature_processing

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'

test = 'Ethanol/IJ Cold'

folder = project_folder + '/Processed/Ethanol'

g = h5py.File(project_folder + '/RawData/Scan_429.hdf5', 'r')

f = h5py.File(project_folder + '/RawData/Scan_423.hdf5', 'r')
q = list(f['q'])

# Grab intensity profiles and crop down to match len(y) (detector collected for too long?)
intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
intensity = intensity[:50]

# Let the last 10 images be the background
bg = intensity[-10:]
bg_avg = np.mean(bg, axis=0)
intensity = [(x-bg_avg) for x in intensity]

# Grab vertical locations
y = np.array(list(f['7bmb1:aero:m1.VAL']))

# Crop down q space to Ethanol calibration range
sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())

# Final processing steps
filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]
reduced_q = np.array(q[sl])
reduced_intensity = [x[sl] for x in filtered_intensity]
reduced_intensity = np.array([y/np.trapz(y, x=reduced_q) for y in reduced_intensity])

temperature_processing(test.rsplit('/')[0], folder, test.rsplit('/')[1], reduced_intensity[:-15], reduced_q, temperature=[], structure_factor=None, y=y[:-15], ramping=False)