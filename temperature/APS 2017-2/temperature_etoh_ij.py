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
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from calc_statistics import comparefit
from temperature_processing import main as temperature_processing

# Plot defaults
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['legend.fontsize'] = 16

def profile(y, parameter, name, profiles_folder, plots_folder):
    np.savetxt(profiles_folder + '/profile_' + name.lower().replace(' ', '') + '.txt', parameter)

    plt.figure()
    plt.plot(y, parameter, ' o', markerfacecolor='none', markeredgecolor='b', label='Data')
    plt.title(test + ' ' + name)
    plt.ylabel(name)
    plt.xlabel('y (mm)')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()
    plt.tick_params(which='both',direction='in')
    plt.tight_layout()
    plt.savefig(plots_folder +  '/' + str(name).lower().replace(' ', '') + '.png')
    plt.close()

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'

tests = ['Ethanol/IJ Ambient', 'Ethanol/IJ Hot']

for test in tests:
    folder = project_folder + '/Processed/Ethanol'
    # profiles_folder = project_folder + '/Processed/' + test + '/Profiles/'
    # if not os.path.exists(profiles_folder):
    #     os.makedirs(profiles_folder)

    # curves_folder = project_folder + '/Processed/' + test + '/Curves/'
    # if not os.path.exists(curves_folder):
    #     os.makedirs(curves_folder)

    # plots_folder = project_folder + '/Processed/' + test + '/Plots/'
    # if not os.path.exists(plots_folder):
    #     os.makedirs(plots_folder)

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
        temperature = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        temperatures.append(np.mean(temperature, axis=0))
    
    sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())
    intensities = [(x-bg_avg) for x in intensities]
    filtered_intensity = [savgol_filter(x, 55, 3) for x in intensities]
    reduced_q = np.array(q[sl])
    reduced_intensity = [x[sl] for x in filtered_intensity]
    reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
    reduced_intensity = [z/np.max(z) for z in reduced_intensity]

    temperature_processing(test.rsplit('/')[0], folder, test.rsplit('/')[1], reduced_intensity, reduced_q, temperature=[], structure_factor=None, y=y, IJ=True)