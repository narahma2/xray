# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:29:08 2019

@author: rahmann
"""

import sys
if sys.platform == 'win32':
	sys.path.append('E:/GitHub/xray/general')
	sys_folder = 'R:/'
elif sys.platform == 'linux':
	sys.path.append('/mnt/e/GitHub/xray/general')
	sys_folder = '/mnt/r/'

import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from calc_statistics import comparefit

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'

test = 'Ethanol Impinging Jet Ambient Temperature Y Scan'
scans = [438, 439, 440, 441, 442, 443, 444]
calib_no = [404, 408, 409]
y = [0.1, 1, 2, 2.5, 4, 15, 25]

g = h5py.File(project_folder + '/RawData/Scan_' + str(437) + '.hdf5', 'r')
bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
bg_avg = np.mean(bg, axis=0)

intensities = []
temperature = []
for scan in scans:
    f = h5py.File(project_folder + '/RawData/Scan_' + str(scan) + '.hdf5', 'r')
    q = list(f['q'])
    intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
    intensities.append(np.mean(intensity, axis=0))
    
sl = slice(43, 450)
intensities = [(x-bg_avg) for x in intensities]
filtered_intensity = [savgol_filter(x, 55, 3) for x in intensities]
reduced_q = q[sl]
reduced_intensity = [x[sl] for x in filtered_intensity]
reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
reduced_intensity = [z/np.max(z) for z in reduced_intensity]

peakq1 = []
peakq2 = []
peakq_ratio = []
var = []
skew = []
peak = []
kurt = []
for j in reduced_intensity:
    peakq1.append(reduced_q[find_peaks(j, distance=100, width=20)[0][0]])
    if len(find_peaks(j, distance=100, width=20)[0]) == 1:
        peakq2.append(np.nan)
    if len(find_peaks(j, distance=100, width=20)[0]) == 2:
        peakq2.append(reduced_q[find_peaks(j, distance=100, width=20)[0][1]])
    peakq_ratio.append(peakq1[-1] / peakq2[-1])
    var.append(np.var(j))
    skew.append(stats.skew(j))
    peak.append(np.max(j))
    kurt.append(stats.kurtosis(j))
    
folder = project_folder + '/Processed/EtOH IJ Ambient'
if not os.path.exists(folder):
    os.makedirs(folder)
    
#etoh_jet_intensity = np.loadtxt(project_folder + '/Processed/Ethanol404/Tests/43_24p78C.txt')
#etoh_jet_q = np.loadtxt(project_folder + '/Processed/Ethanol404/q_range.txt')

#etoh_jet_intensity = np.loadtxt(project_folder + '/Processed/Ethanol408/Tests/03_25p29C.txt')
#etoh_jet_q = np.loadtxt(project_folder + '/Processed/Ethanol408/q_range.txt')

etoh_jet_intensity = np.loadtxt(project_folder + '/Processed/Ethanol409/Tests/77_24p83C.txt')
etoh_jet_q = np.loadtxt(project_folder + '/Processed/Ethanol409/q_range.txt')
    
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['legend.fontsize'] = 16
    
plt.figure()
plt.plot(etoh_jet_q, etoh_jet_intensity / np.max(etoh_jet_intensity), linewidth=2.0, label='EtOH 25°C Jet')
plt.plot(reduced_q, reduced_intensity[1] / np.max(reduced_intensity[1]), linewidth=2.0, linestyle='--', label='y = ' + str(y[1]) + ' mm')
plt.plot(reduced_q, reduced_intensity[4] / np.max(reduced_intensity[1]), linewidth=2.0, linestyle='-.', label='y = ' + str(y[4]) + ' mm')
plt.plot(reduced_q, reduced_intensity[5] / np.max(reduced_intensity[1]), linewidth=2.0, linestyle=':', label='y = ' + str(y[5]) + ' mm')
plt.title('Vertical Variation - 25°C')
plt.legend()
plt.ylabel('Intensity (arb. units)')
plt.xlabel('q (Å$^{-1}$)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(folder + '/vertical_curves.png')

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(y, skew, 'g-o', label='Skewness')
ax2.plot(y, kurt, 'b-s', label='Kurtosis')
ax1.set_xlabel('Vertical Location (mm)')
ax1.set_ylabel('Skewness', color='g')
ax2.set_ylabel('Kurtosis', color='b')
ax1.minorticks_on()
ax2.minorticks_on()
plt.autoscale(enable=True, axis='x')
plt.title('Moments vs. Vertical Location')
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(folder + '/moments.png')

#plt.figure()
#plt.plot(y, kurt, '-[]', color='b', markerfacecolor='b', markeredgecolor='b', label='Kurtosis')
#plt.title('Statistical Moments vs. Vertical Variation')
#plt.ylabel('Kurtosis')
#plt.xlabel('Vertical Location (mm)')
#plt.gca().set_ylim([-1.4, -1.0])
#plt.legend()
##plt.autoscale(enable=True, axis='x', tight=True)
#plt.minorticks_on()
#plt.tick_params(which='both',direction='in')
#plt.tight_layout()
#plt.savefig('E:/APS/Temperature/2017-2/EtOH IJ Ambient/kurtosis.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    