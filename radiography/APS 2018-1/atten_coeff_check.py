# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:16:42 2019

@author: rahmann
"""

import sys
sys.path.append('E:/OneDrive - purdue.edu/Research/GitHub/coding/python')

import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from PIL import Image
from Spectra.spectrum_modeling import multi_angle as xop, xcom, xcom_reshape, density_KIinH2O, beer_lambert, visible_light
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
from skimage.transform import rotate
from Statistics.CIs_LinearRegression import lin_fit

def calc_error(processed_data):
    axial_positions = [x["Axial Position"] for x in processed_data]
    axial_positions = [x for x in axial_positions for x in x]
    
    lateral_positions = [x["Lateral Position"] for x in processed_data]
    lateral_positions = [x for x in lateral_positions for x in x]
    
    optical_diameters = [x["Optical Diameter"] for x in processed_data]
    optical_diameters = [x for x in optical_diameters for x in x]
    
    model_epl = [x["Model EPL Diameter"] for x in processed_data]
    model_epl = [x for x in model_epl for x in x]
        
    summary = pd.DataFrame({"Axial Position": axial_positions, "Lateral Position": lateral_positions, 
                            "Optical Diameter": optical_diameters, "Model EPL Diameter": model_epl})
    
    summary = summary.sort_values(["Axial Position", "Lateral Position"])
    summary = summary.reset_index(drop=True)
    summary_list = [v for k, v in summary.groupby("Axial Position")]
    
    optical_diameters_avg = [np.mean(x["Optical Diameter"]) for x in summary_list][10:]
    model_epl_avg = [np.mean(x["Model EPL Diameter"]) for x in summary_list][10:]
    
    error_avg = abs(np.array(optical_diameters_avg) - np.array(model_epl_avg))
    
    return error_avg

#%% 2000um CF using 700um as calibration
folder = 'R:\\APS 2018-1\\Imaging\\Processed\\Analyzed_Aug23'
tests = glob.glob(folder + "\\*2000*.pckl")

KIconc = [0.0, 1.6, 3.4, 4.8, 8.0, 10.0, 11.1]
error2000_summary = []

plt.figure()
for i, test in enumerate(tests):
    f = open(test, "rb")
    processed_data, positions, cf = pickle.load(f)
    f.close()
    
    error_avg = calc_error(processed_data)
    error2000_summary.append((10*1000*np.mean(error_avg), 10*1000*np.std(error_avg), np.std(error_avg) / np.mean(error_avg)))
    
    plt.subplot(1,2,1)
    plt.plot(positions, savgol_filter(cf, 95, 5), label=str(KIconc[i]) + '% KI')
    plt.subplot(1,2,2)
    plt.plot(positions, 10*1000*np.array(savgol_filter(error_avg, 95, 5)), label=str(KIconc[i]) + '% KI')
    
plt.subplot(1,2,1)
plt.xlabel('Vertical Position')
plt.ylabel('CF [-]')
plt.legend()
plt.title('Final Correction Factor for 2000 $\mu$m')
plt.subplot(1,2,2)
plt.xlabel('Vertical Position')
plt.ylabel('Error [$\mu$m]')
plt.legend()
plt.title('Errors for 2000 $\mu$m')

#%% 700um CF using 2000um as calibration
folder = 'R:\\APS 2018-1\\Imaging\\Processed\\Analyzed_Aug23'
tests = glob.glob(folder + "\\*700*.pckl")

KIconc = [0.0, 1.6, 3.4, 4.8, 8.0, 10.0, 11.1]
error700_summary = []

plt.figure()
for i, test in enumerate(tests):
    f = open(test, "rb")
    processed_data, positions, cf = pickle.load(f)
    f.close()

    error_avg = calc_error(processed_data)
    error700_summary.append((10*1000*np.mean(error_avg[17:-13]), 10*1000*np.std(error_avg[17:-13]), np.std(error_avg[17:-13]) / np.mean(error_avg[17:-13])))
    
    plt.subplot(1,2,1)
    plt.plot(positions[10:-5], savgol_filter(cf[10:-5], 95, 5), label=str(KIconc[i]) + '% KI')
    plt.subplot(1,2,2)
    plt.plot(positions[17:-13], 10*1000*np.array(savgol_filter(error_avg, 23, 3)[17:-13]), label=str(KIconc[i]) + '% KI')
    
plt.subplot(1,2,1)
plt.xlabel('Vertical Position')
plt.ylabel('CF [-]')
plt.legend()
plt.title('Final Correction Factor for 700 $\mu$m')
plt.subplot(1,2,2)
plt.xlabel('Vertical Position')
plt.ylabel('Error [$\mu$m]')
plt.legend()
plt.title('Errors for 700 $\mu$m')















