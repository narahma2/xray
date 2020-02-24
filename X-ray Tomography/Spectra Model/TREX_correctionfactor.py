# -*- coding: utf-8 -*-
"""
Creates the corrected models to be used for final intensity to EPL conversion.
Creates the "50p0_KI_CorrectedModels" folder under /Modeling.

Spectra Modeling Workflow
-------------------------
tube_model -> TREX_calibration -> TREX_correctionfactor -> TREX_error -> TREX_summary

Created on Wed Sep 25 11:06:06 2019

@author: rahmann
"""

import sys
sys.path.append('E:/General Scripts/python')

import pickle
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

#%% SCINTILLATOR PLATE NAMES ARE CORRECTED IN THIS SUMMARY VERSION!
data_folder = 'R:/X-ray Tomography/Spectra Model/ProcessedData/Analyzed'
analyzed_tests = glob.glob(data_folder + '/*.pckl')

summary = pd.DataFrame(columns = ['Run', 'X-ray Tube', 'Scintillator', 'Jet', 'T CF', 'T CF Stats', 'Positions'])

for i, run in enumerate(analyzed_tests):
    f = open(run, "rb")
    [_, positions, T_cf] = pickle.load(f)
    f.close()
    
    ind = int(run.rsplit('\\')[-1].rsplit('_')[0].rsplit('Test')[-1])
    scintillator = run.rsplit('\\')[-1].rsplit('_')[1]
    
    # Correct scintillator name for ACS/ALS
    if scintillator == 'ACS':
        scintillator = 'ALS'
    elif scintillator == 'ALS':
        scintillator = 'ACS'
    
    jet = run.rsplit('\\')[-1].rsplit('_')[2]
    
    if 1 <= ind <= 11:
        xray_tube = 1
    elif 12 <= ind <= 20:
        xray_tube = 2
    elif 21 <= ind <= 34:
        xray_tube = 3
        
    T_cf_mean = np.mean(T_cf)
    T_cf_std = np.std(T_cf)
        
    test_dict = {'Run': ind, 'X-ray Tube': xray_tube, 'Scintillator': scintillator, 'Jet': jet, 'T CF': np.array(T_cf), 
                 'T CF Stats': (T_cf_mean, T_cf_std), 'Positions': np.array(positions)}
    summary.loc[len(summary)] = test_dict
    
#%% Statistic calculations
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
    
#%% Save the corrected models per tube & scintillator combination
f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_model.pckl", "rb")
# Units are in mm!
[cs, spray_epl, RawTransmission] = pickle.load(f)
f.close()

# X-ray Tube Source 1
T_x1_acs = np.array(RawTransmission) / np.mean(x1_acs_mean)
T_x1_als = np.array(RawTransmission) / np.mean(x1_als_mean)
T_x1_fos = np.array(RawTransmission) / np.mean(x1_fos_mean)
cs_x1_acs = CubicSpline(T_x1_acs[::-2], spray_epl[::-2])
cs_x1_als = CubicSpline(T_x1_als[::-2], spray_epl[::-2])
cs_x1_fos = CubicSpline(T_x1_fos[::-2], spray_epl[::-2])

f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x1_acs.pckl", "wb")
pickle.dump([cs_x1_acs, spray_epl, T_x1_acs], f)
f.close()
    
f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x1_als.pckl", "wb")
pickle.dump([cs_x1_als, spray_epl, T_x1_als], f)
f.close()

f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x1_fos.pckl", "wb")
pickle.dump([cs_x1_fos, spray_epl, T_x1_fos], f)
f.close()

# X-ray Tube Source 2
T_x2_acs = np.array(RawTransmission) / np.mean(x2_acs_mean)
T_x2_als = np.array(RawTransmission) / np.mean(x2_als_mean)
T_x2_fos = np.array(RawTransmission) / np.mean(x2_fos_mean)
cs_x2_acs = CubicSpline(T_x2_acs[::-2], spray_epl[::-2])
cs_x2_als = CubicSpline(T_x2_als[::-2], spray_epl[::-2])
cs_x2_fos = CubicSpline(T_x2_fos[::-2], spray_epl[::-2])

f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x2_acs.pckl", "wb")
pickle.dump([cs_x2_acs, spray_epl, T_x2_acs], f)
f.close()
    
f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x2_als.pckl", "wb")
pickle.dump([cs_x2_als, spray_epl, T_x2_als], f)
f.close()

f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x2_fos.pckl", "wb")
pickle.dump([cs_x2_fos, spray_epl, T_x2_fos], f)
f.close()

# X-ray Tube Source 3
T_x3_acs = np.array(RawTransmission) / np.mean(x3_acs_mean)
T_x3_als = np.array(RawTransmission) / np.mean(x3_als_mean)
T_x3_fos = np.array(RawTransmission) / np.mean(x3_fos_mean)
cs_x3_acs = CubicSpline(T_x3_acs[::-2], spray_epl[::-2])
cs_x3_als = CubicSpline(T_x3_als[::-2], spray_epl[::-2])
cs_x3_fos = CubicSpline(T_x3_fos[::-2], spray_epl[::-2])

f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x3_acs.pckl", "wb")
pickle.dump([cs_x3_acs, spray_epl, T_x3_acs], f)
f.close()
    
f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x3_als.pckl", "wb")
pickle.dump([cs_x3_als, spray_epl, T_x3_als], f)
f.close()

f = open("R:/X-ray Tomography/Spectra Model/Modeling/50p0_KI_CorrectedModels/x3_fos.pckl", "wb")
pickle.dump([cs_x3_fos, spray_epl, T_x3_fos], f)
f.close()