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
    sys.path.append('/mnt/e/GitHub/xray/general')
    sys_folder = '/mnt/r/'

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from Form_Factor.xray_factor import ItoS
from temperature_processing import main as temperature_processing

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2/'

# Water: 400, 401, 403
# Ethanol: 404, 408, 409
# Dodecane: 414, 415
scans = [400, 401, 403, 404, 408, 409, 414, 415]
tests = ['Water','Water','Water', 'Ethanol','Ethanol','Ethanol', 'Dodecane','Dodecane']

def main(test, scan):
    folder = project_folder + 'Processed/' + test
    
    f = h5py.File(project_folder + '/RawData/Scan_' + str(scan) + '.hdf5', 'r')
    temperature = list(f['7bm_dau1:dau:010:ADC'])
    temperature.append(temperature[-1])
    q = list(f['q'])
    
    #%% 2017 Water
    if scan == 400:
        intensity = [f['Intensity_vs_q_BG_Sub'][:,i] for i in range(np.shape(f['Intensity_vs_q_BG_Sub'])[1])]
#        sl = slice(6,427)
        sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
        avg_rows = 20
        mid_temp = 109
        
    if scan == 401:
        intensity = [f['Intensity_vs_q_BG_Sub'][:,i] for i in range(np.shape(f['Intensity_vs_q_BG_Sub'])[1])]
#        sl = slice(6,427)
        sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
        avg_rows = 20
        mid_temp = 40
        
    if scan == 403:
        intensity = [f['Intensity_vs_q_BG_Sub'][:,i] for i in range(np.shape(f['Intensity_vs_q_BG_Sub'])[1])]
#        sl = slice(6,427)
        sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
        avg_rows = 20
        mid_temp = 25
        
    #%% 2017 Ethanol
    if (test == 'Ethanol' and scan == 404):
        intensity = [f['Intensity_vs_q_BG_Sub'][:,i] for i in range(np.shape(f['Intensity_vs_q_BG_Sub'])[1])]
        sl = slice(0,195)
        avg_rows = 20
        mid_temp = 71
        
    if scan == 408:
        g = h5py.File(project_folder + '/RawData/Scan_410.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        intensity = [(x-bg_avg) for x in raw_intensity]
        sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())
        avg_rows = 30
        mid_temp = 5
        
    if scan == 409:
        g = h5py.File(project_folder + '/RawData/Scan_410.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        intensity = [(x-bg_avg) for x in raw_intensity]
        sl = slice(43, 452)
        avg_rows = 5
        mid_temp = 54
        
    #%% 2017 Dodecane
    if scan == 414:
        g = h5py.File(project_folder + '/RawData/Scan_416.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        intensity = [(x-bg_avg) for x in raw_intensity]
        sl = slice(43, 452)
        avg_rows = 24
        mid_temp = 6
        
    if scan == 415:
        g = h5py.File(project_folder + '/RawData/Scan_416.hdf5', 'r')
        bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
        intensity = [(x-bg_avg) for x in raw_intensity]
        sl = slice(43, 452)
        avg_rows = 6
        mid_temp = 34
    
    #%% Intensity correction
    if not isinstance(scan, str):
        intensity_avg = []
        temperature_avg = []
        
        for i in range(0, len(intensity) // avg_rows):
            intensity_avg.append(np.mean(intensity[(i*avg_rows):((i+1)*avg_rows-1)], axis=0))
            temperature_avg.append(np.mean(temperature[(i*avg_rows):((i+1)*avg_rows-1)]))
        
        i = (len(intensity) // avg_rows)
        if np.mod(len(intensity), avg_rows) != 0:
            intensity_avg.append(np.mean(intensity[(i*avg_rows):-1], axis=0))
            temperature_avg.append(np.mean(temperature[(i*avg_rows):-1]))
            
        filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity_avg]
        
    elif isinstance(scan, str):
        filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]
        temperature_avg = temperature
        
    reduced_q = np.array(q[sl])
    reduced_intensity = [x[sl] for x in filtered_intensity]
    reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
    reduced_intensity /= np.max(reduced_intensity)
    
    if test == 'Water':
        structure_factor = [ItoS(np.array(reduced_q), x) for x in reduced_intensity]
    else:
        structure_factor = None
    
    temperature_processing(test, folder, scan, reduced_intensity, reduced_q, temperature_avg, structure_factor)
    
    
    rr = np.array([(x-min(temperature))/(max(temperature)-min(temperature)) for x in temperature])
    bb = np.array([1-(x-min(temperature))/(max(temperature)-min(temperature)) for x in temperature])
    
    if 'Water' in test:
        llim = 0.46
    elif 'Ethanol' in test:
        llim = 0.30
    elif 'Dodecane' in test:
        llim = 0.10
        
    plots_folder = folder + '/' + str(scan) + '/Plots/'
    
    plt.figure()
    plt.plot(reduced_q, reduced_intensity[0], linestyle='-', color=(rr[0],0,bb[0]), linewidth=2.0, label=str(int(round(temperature_avg[0],1))) + '°C')
    plt.plot(reduced_q, reduced_intensity[mid_temp], linestyle='-.', color=(rr[mid_temp],0,bb[mid_temp]), linewidth=2.0, label=str(int(round(temperature_avg[mid_temp],1))) + '°C')
    plt.plot(reduced_q, reduced_intensity[-1], linestyle=':', color=(rr[-1],0,bb[-1]), linewidth=2.0, label=str(int(round(temperature_avg[-1],1))) + '°C')
    plt.legend()
    plt.xlabel('q (Å$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.gca().set_ylim([llim, 1.02])
    plt.minorticks_on()
    plt.tick_params(which='both',direction='in')
    plt.title('Select ' + test + ' Curves')
    plt.tight_layout()
    plt.savefig(plots_folder + 'selectcurves.png')
    plt.close()
        
[main(tests[i], scans[i]) for i,_ in enumerate(scans)]
    
    
    
    
    
    
    
    
    
    
    
    
    