# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:39:37 2020

@author: rahmann
"""

import sys
sys.path.append('E:/General Scripts/python')

import pickle
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import rotate
from scipy.signal import savgol_filter, find_peaks, peak_widths

from Spectra.spectrum_modeling import density_KIinH2O
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse

#%% Load inputs
f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_model.pckl", "rb")
# Units are in mm!
[cs, spray_epl, Transmission_50KI] = pickle.load(f)
f.close()

density = density_KIinH2O(50) / 1000
KIperc = 50

atten_50KI = -np.log(np.array(Transmission_50KI))/np.array(spray_epl)/density

# Corresponding transmission value for a given EPL
epl_to_T50KI = CubicSpline(spray_epl, Transmission_50KI)

#%% Experimental setup
mm_pix = np.array([0.04456489, 0.031759657, 0.039928776])
rot = [1.56, -6.27, 6.19]
crop = [slice(500,1000), slice(100,600), slice(300,800)]

data_folder = 'F:/X-ray Tomography/3DXray-Tomo/4Feb2018-Tomo/3DXray_Cam'
corrected = 3 * [None]
epl = 3 * [None]
camera = 3 * [None]

for i, _ in enumerate(mm_pix):
    cap = np.array(Image.open(data_folder + str(i+1) + '/AVG_Cam' + str(i+1) + '_CAP.tif'))
    flat = np.array(Image.open(data_folder + str(i+1) + '/AVG_Cam' + str(i+1) + '_IJ_cant_flat_1.tif'))
    dark = np.array(Image.open(data_folder + str(i+1) + '/AVG_Cam' + str(i+1) + '_IJ_cant_dark_1.tif'))
    
    norm = (cap-dark) / (flat-dark)
    norm[np.isnan(norm)] = 0
    norm[np.isinf(norm)] = 0
    
    corrected[i] = rotate(norm, rot[i])[300:750, crop[i]]
    epl[i] = np.zeros(corrected[i].shape)
    
    for j in range(len(corrected[i])):
        epl[i][j,:] = cs(corrected[i][j,:])
        
    offset_epl = np.median(epl[i][:,0:100])
    epl[i] -= offset_epl
    
    left_bound = len(epl[i]) * [np.nan]
    right_bound = len(epl[i]) * [np.nan]
    model_epl = len(epl[i]) * [np.nan]
    optical_diameter = len(epl[i]) * [np.nan]
    axial_position = len(epl[i]) * [np.nan]
    lateral_position = len(epl[i]) * [np.nan]
    fitted_graph = len(epl[i]) * [np.nan]
    epl_graph = len(epl[i]) * [np.nan]
    actual_diameter = np.zeros((len(epl[i]),1))
    fitted_diameter = np.zeros((len(epl[i]),1))
    epl_corr_factor = np.zeros((len(epl[i]),1))
    trans_corr_factor = np.zeros((len(epl[i]),1))
    for z in range(len(epl[i])):
        warnings.filterwarnings("ignore")
        peaks, _ = find_peaks(savgol_filter(epl[i][z, :], 105, 7), width=20, prominence=0.2)
        warnings.filterwarnings("default")
        
        if len(peaks) == 1:
            warnings.filterwarnings("ignore")
            [full_width, fifth_max, lpos, rpos] = peak_widths(savgol_filter(epl[i][z, :], 105, 7), peaks, rel_height=0.8)
            warnings.filterwarnings("default")
            full_width = full_width[0]
            fifth_max = fifth_max[0]
            left_bound[z] = lpos[0]
            right_bound[z] = rpos[0]
            warnings.filterwarnings("ignore")
            model_epl[z], fitted_graph[z], epl_graph[z] = ideal_ellipse(y=savgol_filter(epl[i][z,:], 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], full_width=full_width, relative_max=fifth_max, dx=mm_pix[i], units='mm')
            warnings.filterwarnings("default")
            optical_diameter[z] = full_width * mm_pix[i]
            axial_position[z] = z
            lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))
            actual_diameter[z] = fitted_graph[z]["a"]*2
            fitted_diameter[z] = fitted_graph[z]["b"]
            epl_corr_factor[z] = actual_diameter[z] / fitted_diameter[z]
            trans_corr_factor[z] = epl_to_T50KI(actual_diameter[z]) / epl_to_T50KI(fitted_diameter[z])
            
    camera[i] = {'EPL CF Mean': np.mean(epl_corr_factor), 'EPL CF StD': np.std(epl_corr_factor), 
                 'T CF Mean': np.mean(trans_corr_factor), 'T CF StD': np.std(trans_corr_factor)}

camera = pd.DataFrame(camera)
    
    
    
    
    
    
    
    
    
    
    
    
    
    