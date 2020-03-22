# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:04:28 2019

@author: rahmann
"""

import sys
sys.path.append("E:/OneDrive - purdue.edu/Research/GitHub/coding/python")

import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy import optimize
from skimage.transform import rotate
from Statistics.CIs_LinearRegression import run_all, lin_fit, conf_calc, ylines_calc, plot_linreg_CIs
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse
from timeit import default_timer as timer

#%% Load models from the whitebeam_2018-1 script
f = open("R:/APS 2018-1/Imaging/water_model.pckl", "rb")
water_model = pickle.load(f)
f.close()

f = open("R:/APS 2018-1/Imaging/KI1p6_model.pckl", "rb")
KI1p6_model = pickle.load(f)
f.close()

f = open("R:/APS 2018-1/Imaging/KI3p4_model.pckl", "rb")
KI3p4_model = pickle.load(f)
f.close()

f = open("R:/APS 2018-1/Imaging/KI4p8_model.pckl", "rb")
KI4p8_model = pickle.load(f)
f.close()

f = open("R:/APS 2018-1/Imaging/KI8p0_model.pckl", "rb")
KI8p0_model = pickle.load(f)
f.close()

f = open("R:/APS 2018-1/Imaging/KI10p0_model.pckl", "rb")
KI10p0_model = pickle.load(f)
f.close()

f = open("R:/APS 2018-1/Imaging/KI11p1_model.pckl", "rb")
KI11p1_model = pickle.load(f)
f.close()

KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]
models = [water_model, KI1p6_model, KI3p4_model, KI4p8_model, KI8p0_model, KI10p0_model, KI11p1_model]

#%% Imaging setup
cm_pix = 700/84 / (1000*10)   # See "APS White Beam.xlsx -> Spatial Resolution"

dark = np.array(Image.open("R:/APS 2018-1/Imaging/Processed/Jets/AVG_Jet_dark2.tif"))

flatfieldavg = np.array(Image.open("R:/APS 2018-1/Imaging/Processed/Jets/AVG_Jet_flat2.tif"))
flatfieldavg_darksub = flatfieldavg - dark

flatfields = glob.glob("R:\\APS 2018-1\\Imaging\\Jets Big and Day\\Jet_flat2\\*.tif")

beam_middle = np.zeros((flatfieldavg_darksub.shape[1]))
for i in range(flatfieldavg_darksub.shape[1]):
    warnings.filterwarnings("ignore")
    beam_middle[i] = np.argmax(savgol_filter(flatfieldavg_darksub[:,i], 55, 3))
    warnings.filterwarnings("default")

beam_middle_avg = int(np.mean(beam_middle).round())

angles = water_model[0]
angle_to_px = [beam_middle_avg+3500*np.tan(x*10**-3)/cm_pix for x in angles]

test_matrix = pd.read_csv("R:/APS 2018-1/Imaging/APS White Beam.txt", sep="\t+", engine="python")
#sl = [1, 8, 15, 44]

for index, _ in enumerate(test_matrix["Test"]):
#for index in sl:
    test_name = test_matrix["Test"][index]
    
    if "2000" in test_name:
        continue
    
    test_path = "R:\\APS 2018-1\\Imaging\\Jets Big and Day\\" + test_name
    tests = glob.glob(test_path + "\\*.tif")
    
    model = models[KI_conc.index(test_matrix["KI %"][index])]
    
    linfits = len(tests) * [None]
    linfits_err = len(tests) * [None]
    r2_values = len(tests) * [None]
    processed_data = len(tests) * [None]
    
    loop_start = timer()
    for i, tt in enumerate(tests):        
        data = np.array(Image.open(tt))
        warnings.filterwarnings("ignore")
        data_norm = (data-dark) / flatfieldavg_darksub
        warnings.filterwarnings("default")
        
        data_epl = np.empty(np.shape(data_norm), dtype=float)
        cropped_view = np.linspace(start=1, stop=340, num=340, dtype=int)
        left_bound = len(cropped_view) * [np.nan]
        right_bound = len(cropped_view) * [np.nan]
        model_epl = len(cropped_view) * [np.nan]
        optical_diameter = len(cropped_view) * [np.nan]
        axial_position = len(cropped_view) * [np.nan]
        lateral_position = len(cropped_view) * [np.nan]
        fitted_graph = len(cropped_view) * [np.nan]
        epl_graph = len(cropped_view) * [np.nan]
        
        for z, k in enumerate(cropped_view):
            j = np.argmin(abs(k-np.array(angle_to_px)))
            data_epl[k, :] = model[1][j](data_norm[k, :])
            
        offset_epl = np.median(data_epl[50:300, 25:125])
        
        #  Change offset EPL to the 700um, 10% case for the hscan tests
        if 'mm' in tt:
            offset_epl = -0.002720126784380998
            
        data_epl -= offset_epl
        
        if "700" in test_name:
            data_epl = rotate(data_epl, 2.0)
            
        for z, k in enumerate(cropped_view):
            warnings.filterwarnings("ignore")
            peaks, _ = find_peaks(savgol_filter(data_epl[k, :], 105, 7), width=20, prominence=0.1)
            warnings.filterwarnings("default")
            
            if len(peaks) == 1:
                warnings.filterwarnings("ignore")
                [full_width, fifth_max, lpos, rpos] = peak_widths(savgol_filter(data_epl[k, :], 105, 7), peaks, rel_height=0.8)
                warnings.filterwarnings("default")
                full_width = full_width[0]
                fifth_max = fifth_max[0]
                left_bound[z] = lpos[0]
                right_bound[z] = rpos[0]
#                model_epl[z], fitted_graph[z], epl_graph[z] = ideal_sphere(y=data_epl[k, int(round(lpos[0])):int(round(rpos[0]))], dx=cm_pix)
                warnings.filterwarnings("ignore")
                model_epl[z], fitted_graph[z], epl_graph[z] = ideal_ellipse(y=savgol_filter(data_epl[k,:], 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], full_width=full_width, relative_max=fifth_max, dx=cm_pix)
                warnings.filterwarnings("default")
                optical_diameter[z] = full_width * cm_pix
                axial_position[z] = k
                lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))
        
        if len(peaks) == 1:
            signal = np.nanmean(data_epl[10:340, int(round(np.nanmean(lateral_position)))-20:int(round(np.nanmean(lateral_position)))+20])
        else:
            signal = 0
            
        noise = np.nanstd(data_epl[50:300, 25:125])
        SNR = signal / noise
        
        try:
            idx = np.isfinite(model_epl) & np.isfinite(optical_diameter)
            linfits[i], linfits_err[i], r2_values[i] = lin_fit(np.array(model_epl)[idx], np.array(optical_diameter)[idx])
            
        except:
            pass
        
        processed_data[i] = {"Image": tt, "Optical Diameter": optical_diameter, "Model EPL Diameter": model_epl, "Axial Position": axial_position,
                             "Lateral Position": lateral_position, "Left Bound": left_bound, "Right Bound": right_bound, "Linear Fits": linfits[i], 
                             "Linear Fits Error": linfits_err[i], "R2": r2_values[i], "Offset EPL": offset_epl, "SNR": SNR}
            
        iteration_time = timer()
        if np.mod(i, 50) == 0:
            print("On Test " + str(index+1) + " of " + str(len(test_matrix["Test"])) + ". " + str(round(100*(i+1)/len(tests))) + "% completed. " + 
                  str(round((iteration_time - loop_start)/60, 2)) + " minutes in. ETA: " + 
                  str(round(((iteration_time - loop_start)/((i+1)/len(tests)) - (iteration_time - loop_start))/60, 2)) + " minutes left.")
            
#        if i == 0:
#            break
    
    f = open("R:/APS 2018-1/Imaging/Processed/Jets/" + test_name + ".pckl", "wb")
    pickle.dump(processed_data, f)
    f.close()
    
#    if index==15:
#        break
#    break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    