# -*- coding: utf-8 -*-
"""
Processes the APS 2017-2 temperature data sets.
Creates the calibration sets to be used for the impinging jets.

Created on Sat Aug 25 14:51:24 2018

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

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.constants import convert_temperature
from Form_Factor.xray_factor import ItoS
from temperature_processing import main as temperature_processing

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2/'

# Water: 400, 401, 403
# Ethanol: 404, 408, 409
# Dodecane: 414, 415
# scans = [397, 398, 400, 401, 403, 404, 408, 409, 414, 415]
# tests = ['Water','Water','Water','Water','Water', 'Ethanol','Ethanol','Ethanol', 'Dodecane','Dodecane']
scans = [400, 401, 403, 408,409, 414, 415]
tests = ['Water','Water','Water', 'Ethanol','Ethanol', 'Dodecane','Dodecane']

def main(test, scan):
    folder = project_folder + 'Processed/' + test
    
    f = h5py.File(project_folder + '/RawData/Scan_' + str(scan) + '.hdf5', 'r')
    temperature = list(f['7bm_dau1:dau:010:ADC'])
    temperature.append(temperature[-1])

    # Convert temperature from Celsius to Kelvin
    temperature = convert_temperature(temperature, 'Celsius', 'Kelvin')

    q = list(f['q'])
    
    #%% 2017 Water
    if scan in [397, 398]:
        g = h5py.File(project_folder + '/RawData/Scan_402.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
        avg_rows = 12

    if scan in [400, 401, 403]:
        g = h5py.File(project_folder + '/RawData/Scan_402.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
        avg_rows = 20
        
    #%% 2017 Ethanol
    # 404 looked at the same q range as the water scans
    if scan == 404:
        g = h5py.File(project_folder + '/RawData/Scan_402.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
        avg_rows = 20
        
    # 408 and 409 had a different detector position to inspect a different q range
    if scan == 408:
        g = h5py.File(project_folder + '/RawData/Scan_410.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())
        avg_rows = 5 
        
    if scan == 409:
        g = h5py.File(project_folder + '/RawData/Scan_410.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())
        avg_rows = 5
        
    #%% 2017 Dodecane
    if scan == 414:
        g = h5py.File(project_folder + '/RawData/Scan_416.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())
        avg_rows = 24
        
    if scan == 415:
        g = h5py.File(project_folder + '/RawData/Scan_416.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())
        avg_rows = 6
    
    # Background subtraction
    intensity = [(x-bg_avg) for x in raw_intensity]
    # intensity = raw_intensity

    #%% Intensity correction
    intensity_avg = []
    temperature_avg = []
    scatter = []
    
    # Bin the data sets
    for i in range(0, len(intensity) // avg_rows):
        scatter.append(np.mean(f['Scatter_images'][(i*avg_rows):((i+1)*avg_rows-1)], axis=0))
        intensity_avg.append(np.mean(intensity[(i*avg_rows):((i+1)*avg_rows-1)], axis=0))
        temperature_avg.append(np.mean(temperature[(i*avg_rows):((i+1)*avg_rows-1)]))
    
    i = (len(intensity) // avg_rows)
    if np.mod(len(intensity), avg_rows) != 0:
        scatter.append(np.mean(f['Scatter_images'][(i*avg_rows):-1], axis=0))
        intensity_avg.append(np.mean(intensity[(i*avg_rows):-1], axis=0))
        temperature_avg.append(np.mean(temperature[(i*avg_rows):-1]))
        
    filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity_avg]
        
    reduced_q = np.array(q[sl])
    reduced_intensity = [x[sl] for x in filtered_intensity]
    reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
    
    if test == 'Water':
        structure_factor = [ItoS(np.array(reduced_q), x) for x in reduced_intensity]
    else:
        structure_factor = None
    
    temperature_processing(test, folder, scan, reduced_intensity, reduced_q, temperature_avg, structure_factor, scatter=scatter, background=g['Scatter_images'])
    
    
    rr = np.array([(x-min(temperature))/(max(temperature)-min(temperature)) for x in temperature])
    bb = np.array([1-(x-min(temperature))/(max(temperature)-min(temperature)) for x in temperature])
    
    if 'Water' in test:
        llim = 0.46
    elif 'Ethanol' in test:
        llim = 0.30
    elif 'Dodecane' in test:
        llim = 0.10
        
    plots_folder = folder + '/' + str(scan) + '/Plots/'
    mid_temp = len(reduced_intensity) // 2  # grab the middle slice to get the median temperature
    plt.figure()
    plt.plot(reduced_q, reduced_intensity[0], linestyle='-', color=(rr[0],0,bb[0]), linewidth=2.0, label=str(int(round(temperature_avg[0],1))) + ' K')
    plt.plot(reduced_q, reduced_intensity[mid_temp], linestyle='-.', color=(0.5,0,0.5), linewidth=2.0, label=str(int(round(temperature_avg[mid_temp],1))) + ' K')
    plt.plot(reduced_q, reduced_intensity[-1], linestyle=':', color=(rr[-1],0,bb[-1]), linewidth=2.0, label=str(int(round(temperature_avg[-1],1))) + ' K')
    plt.legend()
    plt.xlabel('q (Å$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.autoscale(enable=True, axis='x', tight=True)
    #plt.gca().set_ylim([llim, 1.02])
    plt.minorticks_on()
    plt.tick_params(which='both',direction='in')
    plt.title('Select ' + test + ' Curves')
    plt.tight_layout()
    plt.savefig(plots_folder + 'selectcurves.png')
    plt.close()
  
# Run all tests      
[main(tests[i], scans[i]) for i,_ in enumerate(scans)]
    
# Run select tests (only ethanol & dodecane)
# [main(tests[i], scans[i]) for i in [j for j,_ in enumerate(tests) if tests[j] == 'Water']] 
