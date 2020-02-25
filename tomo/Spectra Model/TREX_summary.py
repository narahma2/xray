# -*- coding: utf-8 -*-
"""
Summarizes the results from the spectra modeling effort.

Spectra Modeling Workflow
-------------------------
tube_model -> TREX_calibration -> TREX_correctionfactor -> TREX_error -> TREX_summary

Created on Fri Sep 27 09:45:36 2019

@author: rahmann
"""

import sys
sys.path.append('E:/General Scripts/python')

import pickle
import glob
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, find_peaks, peak_widths
from Spectra.spectrum_modeling import density_KIinH2O
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse

#%%
data_folder = 'R:/X-ray Tomography/Spectra Model/ProcessedData/Corrected/'
datasets = glob.glob(data_folder + '*.pckl')

summary = pd.DataFrame(columns = ['Run', 'X-ray Tube', 'Scintillator', 'Jet', 'KI%', 'Error', '% Error', 'Error Mean', 'Error StDev', 'Positions'])

for run in datasets:
    f = open(run, "rb")
    [processed_data, positions, error_summary] = pickle.load(f)
    f.close()
    
    ind = int(run.rsplit('\\')[-1].rsplit('_')[0].rsplit('Test')[-1])
    scintillator = run.rsplit('\\')[-1].rsplit('_')[1]
    
    jet = run.rsplit('\\')[-1].rsplit('_')[2]
    KIperc = run.rsplit('\\')[-1].rsplit('_')[3].rsplit('KI')[0]
    
    if 1 <= ind <= 11:
        xray_tube = 1
    elif 12 <= ind <= 20:
        xray_tube = 2
    elif 21 <= ind <= 34:
        xray_tube = 3
        
    # Calculate error
    axial_positions = [x["Axial Position"] for x in processed_data]
    axial_positions = [x for x in axial_positions for x in x]
    
    lateral_positions = [x["Lateral Position"] for x in processed_data]
    lateral_positions = [x for x in lateral_positions for x in x]
    
    optical_diameters = [x["Optical Diameter"] for x in processed_data]
    optical_diameters = [x for x in optical_diameters for x in x]
    
    model_epl = [x["Model EPL Diameter"] for x in processed_data]
    model_epl = [x for x in model_epl for x in x]
    
    SNR = np.mean([x["SNR"] for x in processed_data])
    
    summ = pd.DataFrame({"Axial Position": axial_positions, "Lateral Position": lateral_positions, 
                            "Optical Diameter": optical_diameters, "Model EPL Diameter": model_epl})
    
    summ = summ.sort_values(["Axial Position", "Lateral Position"])
    summ = summ.reset_index(drop=True)
    summ_list = [v for k, v in summ.groupby("Axial Position")]
    
    positions = [np.mean(x["Axial Position"]) for x in summ_list][10:]
    optical_diameters_avg = [np.mean(x["Optical Diameter"]) for x in summ_list]
    model_epl_avg = [np.mean(x["Model EPL Diameter"]) for x in summ_list]
    # Absolute error
    error_avg = abs(np.array(optical_diameters_avg) - np.array(model_epl_avg))[10:]
    # Percent error
    error_perc = np.array(error_avg) / np.array(optical_diameters_avg[10:])
        
    test_dict = {'Run': ind, 'X-ray Tube': xray_tube, 'Scintillator': scintillator, 'Jet': jet, 'KI%': KIperc, 'Error': error_avg,
                 '% Error': error_perc, 'Error Mean': error_summary[0], 'Error StDev': error_summary[1], 'Positions': np.array(positions)}
    summary.loc[len(summary)] = test_dict
        
#%% Statistic summary
# Jet summary (per tube & scintillator)
x1_acs_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 1) & (summary["Scintillator"] == "ACS")]["T CF"].index]])
x2_acs_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 2) & (summary["Scintillator"] == "ACS")]["T CF"].index]])
x3_acs_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 3) & (summary["Scintillator"] == "ACS")]["T CF"].index]])
x1_acs_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 1) & (summary["Scintillator"] == "ACS")]["T CF"].index]])
x2_acs_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 2) & (summary["Scintillator"] == "ACS")]["T CF"].index]])
x3_acs_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 3) & (summary["Scintillator"] == "ACS")]["T CF"].index]])

x1_als_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 1) & (summary["Scintillator"] == "ALS")]["T CF"].index]])
x2_als_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 2) & (summary["Scintillator"] == "ALS")]["T CF"].index]])
x3_als_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 3) & (summary["Scintillator"] == "ALS")]["T CF"].index]])
x1_als_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 1) & (summary["Scintillator"] == "ALS")]["T CF"].index]])
x2_als_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 2) & (summary["Scintillator"] == "ALS")]["T CF"].index]])
x3_als_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 3) & (summary["Scintillator"] == "ALS")]["T CF"].index]])

x1_fos_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 1) & (summary["Scintillator"] == "FOS")]["T CF"].index]])
x2_fos_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 2) & (summary["Scintillator"] == "FOS")]["T CF"].index]])
x3_fos_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 3) & (summary["Scintillator"] == "FOS")]["T CF"].index]])
x1_fos_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 1) & (summary["Scintillator"] == "FOS")]["T CF"].index]])
x2_fos_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 2) & (summary["Scintillator"] == "FOS")]["T CF"].index]])
x3_fos_std = np.std([summary["T CF"][y] for y in [x for x in summary[(summary["X-ray Tube"] == 3) & (summary["Scintillator"] == "FOS")]["T CF"].index]])

source_scint_summary = np.array([[x1_acs_mean, x1_acs_std], [x1_als_mean, x1_als_std], [x1_fos_mean, x1_fos_std],
                                 [x2_acs_mean, x2_acs_std], [x2_als_mean, x2_als_std], [x2_fos_mean, x2_fos_std], 
                                 [x3_acs_mean, x3_acs_std], [x3_als_mean, x3_als_std], [x3_fos_mean, x3_fos_std]])

# Scintillator summary (per scintillator)
acs_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[summary["Scintillator"] == "ACS"]["T CF"].index]])
als_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[summary["Scintillator"] == "ALS"]["T CF"].index]])
fos_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[summary["Scintillator"] == "FOS"]["T CF"].index]])
acs_std = np.std([summary["T CF"][y] for y in [x for x in summary[summary["Scintillator"] == "ACS"]["T CF"].index]])
als_std = np.std([summary["T CF"][y] for y in [x for x in summary[summary["Scintillator"] == "ALS"]["T CF"].index]])
fos_std = np.std([summary["T CF"][y] for y in [x for x in summary[summary["Scintillator"] == "FOS"]["T CF"].index]])

scint_summary = np.array([[acs_mean, acs_std], [als_mean, als_std], [fos_mean, fos_std]])

# Tube source summary (per tube)
x1_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[summary["X-ray Tube"] == 1]["T CF"].index]])
x2_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[summary["X-ray Tube"] == 2]["T CF"].index]])
x3_mean = np.mean([summary["T CF"][y] for y in [x for x in summary[summary["X-ray Tube"] == 3]["T CF"].index]])
x1_std = np.std([summary["T CF"][y] for y in [x for x in summary[summary["X-ray Tube"] == 1]["T CF"].index]])
x2_std = np.std([summary["T CF"][y] for y in [x for x in summary[summary["X-ray Tube"] == 2]["T CF"].index]])
x3_std = np.std([summary["T CF"][y] for y in [x for x in summary[summary["X-ray Tube"] == 3]["T CF"].index]])

tube_summary = np.array([[x1_mean, x1_std], [x2_mean, x2_std], [x3_mean, x3_std]])        
        
        
        
        
        
        
        
        
        
        
    