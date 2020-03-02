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
from temperature_processing import main as temperature_processing

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'

test = 'Ethanol/IJ Cold'

folder = project_folder + '/Processed/Ethanol'

g = h5py.File(project_folder + '/RawData/Scan_427.hdf5', 'r')
bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
bg_avg = np.mean(bg, axis=0)

f = h5py.File(project_folder + '/RawData/Scan_423.hdf5', 'r')
q = list(f['q'])
intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
intensity = [(x-bg_avg) for x in intensity]
y = list(f['7bmb1:aero:m1.VAL'])    
    
sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())

filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]
reduced_q = np.array(q[sl])
reduced_intensity = [x[sl] for x in filtered_intensity]
reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
reduced_intensity = [z/np.max(z) for z in reduced_intensity]

temperature_processing(test.rsplit('/')[0], folder, test.rsplit('/')[1], reduced_intensity, reduced_q, temperature=[], structure_factor=None, y=y, IJ=True)