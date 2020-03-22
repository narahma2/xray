# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:36:32 2019

@author: rahmann
"""

import sys
sys.path.append('E:/OneDrive - purdue.edu/Research/GitHub/coding/python')

import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy import optimize
from Statistics.CIs_LinearRegression import run_all, lin_fit, conf_calc, ylines_calc, plot_linreg_CIs
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse
from timeit import default_timer as timer

#%% Load models from the whitebeam_2019-1 script
f = open("R:/APS 2019-1/KI4p8_model.pckl", "rb")
KI4p8_model = pickle.load(f)
f.close()
#%% Imaging setup
cm_pix = 20 / 2.1 / (1000*10)

dark = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_dark_current.tif'))
flatfield = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_Background_NewYAG.tif'))
flatfield_darksub = flatfield - dark

beam_middle = np.zeros((flatfield_darksub.shape[1]))
for i in range(flatfield_darksub.shape[1]):
    beam_middle[i] = np.argmax(savgol_filter(flatfield_darksub[:,i], 55, 3))

beam_middle_avg = int(np.mean(beam_middle).round())

angles = KI4p8_model[0]
angle_to_px = [beam_middle_avg+3500*np.tan(x*10**-3)/cm_pix for x in angles]

test_matrix = pd.read_csv("R:/APS 2019-1/test_matrix.txt", sep="\t+", engine="python")

cal_ind = [a for a, b in enumerate(["KI_Liquid_Jet" in c for c in test_matrix["Test"]]) if b]

for index in cal_ind:
    test_path = "R:\\APS 2019-1\\Imaging\\Raw_WB_Images\\" + test_matrix["Test"][index]
    tests = glob.glob(test_path + "\\*.tif")
    
    processed_data = len(tests) * [None]
    
    loop_start = timer()
    for i, tt in enumerate(tests):
        data = np.array(Image.open(tt))
        data_norm = (data-dark) / flatfield_darksub
        
        data_epl = np.empty(np.shape(data_norm), dtype=float)
        cropped_view = np.linspace(start=test_matrix["Cropping Start"][index], stop=480, num=480-test_matrix["Cropping Start"][index]+1, dtype=int)
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
            data_epl[k, :] = KI4p8_model[1][j](data_norm[k, :])
            
        roi = '215:350, 640:740'
        offset_epl = np.median(data_epl[int(roi.rsplit(",")[0].rsplit(":")[0][1:]):int(roi.rsplit(",")[0].rsplit(":")[1]), int(roi.rsplit(",")[1].rsplit(":")[0]):int(roi.rsplit(",")[1].rsplit(":")[1][:-1])])
        data_epl -= - offset_epl
        
        for z, k in enumerate(cropped_view):
            warnings.filterwarnings("ignore")
            peaks, _ = find_peaks(savgol_filter(data_epl[k, :], 105, 7), width=100)
            warnings.filterwarnings("default")     
            
            if len(peaks) == 1:
                [full_width, half_max, lpos, rpos] = peak_widths(savgol_filter(data_epl[k, :], 105, 7), peaks, rel_height=0.8)
                full_width = full_width[0]
                half_max = half_max[0]
                left_bound[z] = lpos[0]
                right_bound[z] = rpos[0]
                model_epl[z], fitted_graph[z], epl_graph[z] = ideal_ellipse(y=savgol_filter(data_epl[k,:], 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], full_width=full_width, relative_max=half_max, dx=cm_pix)
                optical_diameter[z] = full_width * cm_pix
                axial_position[z] = k
                lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))
                
        noise = np.nanstd(data_epl[int(roi.rsplit(",")[0].rsplit(":")[0][1:]):int(roi.rsplit(",")[0].rsplit(":")[1]), int(roi.rsplit(",")[1].rsplit(":")[0]):int(roi.rsplit(",")[1].rsplit(":")[1][:-1])])
        if len(peaks) == 1:
            signal = np.median(data_epl[:, int(round(np.nanmean(lateral_position)))-20:int(round(np.nanmean(lateral_position)))+20])
        else:
            signal = 0
        SNR = signal / noise        
        
        processed_data[i] = {"Image": tt, "Optical Diameter": optical_diameter, "Model EPL Diameter": model_epl, "Axial Position": axial_position,
                             "Lateral Position": lateral_position, "Left Bound": left_bound, "Right Bound": right_bound, "Offset EPL": offset_epl, "SNR": SNR}
        
        iteration_time = timer()
        if np.mod(i, 50) == 0:
            print("On Test " + str(cal_ind.index(index)+1) + " of " + str(len(cal_ind)) + " (" + test_path.rsplit("Raw_WB_Images\\")[1] + "). " + 
                  str(round(100*(i+1)/len(tests))) + "% completed. " + str(round((iteration_time - loop_start)/60, 2)) + " minutes in. ETA: " + 
                  str(round(((iteration_time - loop_start)/((i+1)/len(tests)) - (iteration_time - loop_start))/60, 2)) + " minutes.")
            
#        if i == 0:
#            break
    
    f = open("R:/APS 2019-1/Imaging/Processed_WB_Images/KI/Calibration/" + test_matrix["Test"][index] + ".pckl", "wb")
    pickle.dump(processed_data, f)
    f.close()

    
    
    
    
    
    
    
    
    