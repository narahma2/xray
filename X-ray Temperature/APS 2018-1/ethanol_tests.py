# -*- coding: utf-8 -*-
"""
Processes the APS 2018-1 ethanol jet data sets.
Created on Thu Jan 23 17:42:32 2020

@author: rahmann
"""

import sys
sys.path.append('E:/General Scripts/python')
sys.path.append('R:/X-ray Temperature/Scripts')

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from Form_Factor.xray_factor import ItoS
from temperature_processing import main as temperature_processing

#%% Setup
project_folder = 'R:/X-ray Temperature/APS 2018-1/'

test = 'Ethanol_700umNozzle'

folder = project_folder + 'Processed/' + test
if not os.path.exists(folder):
    os.makedirs(folder)
    
scans = ['/RampUp', '/RampDown', '/Combined']

for scan in scans:
    #%% Load background
    bg = glob.glob(project_folder + '/Q Space/Ethanol_700umNozzle/*Scan1112*')
    q = np.loadtxt(bg[0], usecols=0)                    # Load q
    bg = [np.loadtxt(x, usecols=1) for x in bg]         # Load intensity
    
    # Average background intensities
    bg = np.mean(bg, axis=0)

    #%% Intensity correction
    rampUp = [1109, 1111, 1113, 1115, 1117, 1118, 1119, 1120]
    rampDown = [1121, 1122, 1124, 1125, 1126, 1128, 1129, 1130]
    combined = rampUp + rampDown
    
    if 'Up' in scan:
        files = rampUp
    elif 'Down' in scan:
        files = rampDown
    else:
        files = combined
        
    files = [glob.glob(project_folder + '/Q Space/Ethanol_700umNozzle/*' + str(x) + '*') for x in files]
    temperature = np.array([float(x[0].rsplit('Target')[-1].rsplit('_')[0]) for x in files])
    
    intensity = [np.mean([np.loadtxt(x, usecols=1)-bg for x in y], axis=0) for y in files]    
            
    filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]
    
    sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())
    
    reduced_q = q[sl]
    reduced_intensity = [x[sl] for x in filtered_intensity]
    reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
    reduced_intensity /= np.max(reduced_intensity)
    
    structure_factor = None

    temperature_processing(test, folder, scan, reduced_intensity, reduced_q, temperature, structure_factor)        
        
        
        
        
        
        
        
        
        
        
        
        