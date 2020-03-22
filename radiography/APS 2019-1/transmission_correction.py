# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:47:15 2019

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
from Statistics.CIs_LinearRegression import lin_fit
        
#%% Load models from the whitebeam_2019-1 script
f = open("R:/APS 2019-1/water_model.pckl", "rb")
water_model = pickle.load(f)
f.close()

f = open("R:/APS 2019-1/KI4p8_model.pckl", "rb")
KI4p8_model = pickle.load(f)
f.close()

#%% Calculate beam attenuation coefficients [1/cm]
density = [1.0, density_KIinH2O(4.8)]
KIperc = [0, 4.8]
spray_epl = water_model[2][1:]

water_model[3] = [water_model[3][i][1:] for i,_ in enumerate(water_model[3])]
KI4p8_model[3] = [KI4p8_model[3][i][1:] for i,_ in enumerate(KI4p8_model[3])]

atten_water = [-np.log(np.array(x))/spray_epl/density[0] for x in water_model[3]]
atten_KI4p8 = [-np.log(np.array(x))/spray_epl/density[1] for x in KI4p8_model[3]]

atten_all = [atten_water, atten_KI4p8]
trans_all = [water_model[3], KI4p8_model[3]]
cs_all = [[CubicSpline(trans_all[0][n][::-1], spray_epl[::-1]) for n,_ in enumerate(trans_all[0])], 
      [CubicSpline(trans_all[1][n][::-1], spray_epl[::-1]) for n,_ in enumerate(trans_all[1])]]


p = len(spray_epl) * [None]
y_err = len(spray_epl) * [None]
r2 = len(spray_epl) * [None]
pT = len(spray_epl) * [None]
y_errT = len(spray_epl) * [None]
r2T = len(spray_epl) * [None]
pfits = len(atten_water) * [None]
pTfits = len(atten_water) * [None]
model_fits = len(atten_water) * [None]  
    
for j, _ in enumerate(atten_water):
    for i, _ in enumerate(spray_epl):
        # Fitting of attenuation coefficient of each model against the KI % of that model
        # Each "i" represents each of the individual EPL values tested
        p[i], y_err[i], r2[i] = lin_fit(KIperc, [y[j][i] for y in atten_all])
        pT[i], y_errT[i], r2T[i] = lin_fit(KIperc, [y[j][i] for y in trans_all])
        
    # Each "j" is the vertical position in the WB. "i" is the individual EPL.
    # pfits[300][70](1.6) : Returns the attenuation coefficient at middle of WB, for EPL = 700 um and KI% = 1.6%
    pfits[j] = [lin_fit(KIperc, [x[j][k] for x in atten_all])[0] for k,_ in enumerate(spray_epl)]
    # pTfits[300][70](1.6) : Returns the transmission at middle of WB, for EPL = 700 um and KI% = 1.6%
    pTfits[j] = [lin_fit(KIperc, [x[j][k] for x in trans_all])[0] for k,_ in enumerate(spray_epl)]
    
    fits = [[x(KIperc)[0] for x in p], [x(KIperc)[1] for x in p]]

    model_fits[j] = [CubicSpline(x[::-1][:-1], spray_epl[::-1][:-1]) for x in fits]
    
#%% Attenuation coefficient correction
cm_pix = 20 / 2.1 / (1000*10)

dark = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_dark_current.tif'))
flatfield = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_Background_NewYAG.tif'))
flatfield_darksub = flatfield - dark

beam_middle = np.zeros((flatfield_darksub.shape[1]))
for i in range(flatfield_darksub.shape[1]):
    beam_middle[i] = np.argmax(savgol_filter(flatfield_darksub[:,i], 55, 3))

beam_middle_avg = int(np.mean(beam_middle).round())

angles = water_model[0]
angle_to_px = [beam_middle_avg+3500*np.tan(x*10**-3)/cm_pix for x in angles]

test_matrix = pd.read_csv("R:/APS 2019-1/test_matrix.txt", sep="\t+", engine="python")

iii = [0, 1]    # Model (water / KI)
jjj = [3, 36]   # Test index

for nn, _ in enumerate(iii):
    ii = iii[nn]
    jj = jjj[nn]
    
    test_name = test_matrix["Test"][jj]
    test_path = "R:\\APS 2019-1\\Imaging\\Raw_WB_Images\\" + test_name
    tests = glob.glob(test_path + "\\*.tif")
    
    actual_diameter = np.zeros((len(tests),1))
    fitted_diameter = np.zeros((len(tests),1))
    cropped_view = np.linspace(start=test_matrix["Cropping Start"][jj], stop=480, num=480-test_matrix["Cropping Start"][jj]+1, dtype=int)

    for i, tt in enumerate(tests):        
        data = np.array(Image.open(tt))
        warnings.filterwarnings("ignore")
        data_norm = (data-dark) / flatfield_darksub
        warnings.filterwarnings("default")
        roi = test_matrix["ROI"][jj]
        offset = 1-np.median(data_norm[int(roi.rsplit(",")[0].rsplit(":")[0][1:]):int(roi.rsplit(",")[0].rsplit(":")[1]), int(roi.rsplit(",")[1].rsplit(":")[0]):int(roi.rsplit(",")[1].rsplit(":")[1][:-1])])

        j = np.argmin(abs(beam_middle_avg-np.array(angle_to_px)))
        data_epl = cs_all[ii][j](data_norm[beam_middle_avg, :] + offset)
        offset_epl = np.median(data_epl[10:100])
        data_epl -= offset_epl
    
        peaks, _ = find_peaks(savgol_filter(data_epl, 105, 7), width=100)
        [full_width, fifth_max, lpos, rpos] = peak_widths(savgol_filter(data_epl, 105, 7), peaks, rel_height=0.8)
        full_width = full_width[0]
        fifth_max = fifth_max[0]
        left_bound = lpos[0]
        right_bound = rpos[0]
    
        model_epl, fitted_graph, epl_graph = ideal_ellipse(y=savgol_filter(data_epl, 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], 
                                                           full_width=full_width, relative_max=fifth_max, dx=cm_pix)
        
        actual_diameter[i] = fitted_graph["a"]*2
        fitted_diameter[i] = fitted_graph["b"]
        
    
    actual_epl = [np.argmin(abs(spray_epl-x)) for x in actual_diameter]
    fitted_epl = [np.argmin(abs(spray_epl-x)) for x in fitted_diameter]
    
    trans_corr_factor = [pTfits[300][actual_epl[i]](KIperc[ii]) / pTfits[300][fitted_epl[i]](KIperc[ii]) for i,_ in enumerate(actual_epl)]
    
    # Cubic spline fitting of Transmission vs. spray_epl curves (needs to be reversed b/c of monotonically increasing
    # restriction on 'x', however this does not change the interpolation call)
    cs = [CubicSpline(np.array(trans_all[ii][n][::-1]) / np.mean(trans_corr_factor), spray_epl[::-1]) for n,_ in enumerate(trans_all[ii])]
            
    #%% Applying the fixed transmission
    actual_diameter_cf = np.zeros((len(tests),1))
    fitted_diameter_cf = np.zeros((len(tests),1))
    cropped_view = np.linspace(start=test_matrix["Cropping Start"][jj], stop=480, num=480-test_matrix["Cropping Start"][jj]+1, dtype=int)

    for i, tt in enumerate(tests):        
        data = np.array(Image.open(tt))
        warnings.filterwarnings("ignore")
        data_norm = (data-dark) / flatfield_darksub
        warnings.filterwarnings("default")
        roi = test_matrix["ROI"][jj]
        offset = 1-np.median(data_norm[int(roi.rsplit(",")[0].rsplit(":")[0][1:]):int(roi.rsplit(",")[0].rsplit(":")[1]), int(roi.rsplit(",")[1].rsplit(":")[0]):int(roi.rsplit(",")[1].rsplit(":")[1][:-1])])

        j = np.argmin(abs(beam_middle_avg-np.array(angle_to_px)))
        data_epl = cs[j](data_norm[beam_middle_avg, :] + offset)
        
        offset_epl = np.median(data_epl[10:100])
        data_epl -= offset_epl
    
        peaks, _ = find_peaks(savgol_filter(data_epl, 105, 7), width=100)
        [full_width, fifth_max, lpos, rpos] = peak_widths(savgol_filter(data_epl, 105, 7), peaks, rel_height=0.8)
        full_width = full_width[0]
        fifth_max = fifth_max[0]
        left_bound = lpos[0]
        right_bound = rpos[0]
    
        model_epl, fitted_graph, epl_graph = ideal_ellipse(y=savgol_filter(data_epl, 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], 
                                                           full_width=full_width, relative_max=fifth_max, dx=cm_pix)
        
        actual_diameter_cf[i] = fitted_graph["a"]*2
        fitted_diameter_cf[i] = fitted_graph["b"]    
        
    actual_epl_cf = [np.argmin(abs(spray_epl-x)) for x in actual_diameter_cf]
    fitted_epl_cf = [np.argmin(abs(spray_epl-x)) for x in fitted_diameter_cf]
    
    trans_corr_factor_cf = [pTfits[300][actual_epl_cf[i]](0) / pTfits[300][fitted_epl_cf[i]](0) for i,_ in enumerate(actual_epl_cf)]   
    
    #%% Calculation of vertical correction factor
    linfits = len(tests) * [None]
    linfits_err = len(tests) * [None]
    r2_values = len(tests) * [None]
    processed_data = len(tests) * [None]
    
    loop_start = timer()
    for i, tt in enumerate(tests):        
        data = np.array(Image.open(tt))
        warnings.filterwarnings("ignore")
        data_norm = (data-dark) / flatfield_darksub
        warnings.filterwarnings("default")
        roi = test_matrix["ROI"][jj]
        offset = 1-np.median(data_norm[int(roi.rsplit(",")[0].rsplit(":")[0][1:]):int(roi.rsplit(",")[0].rsplit(":")[1]), int(roi.rsplit(",")[1].rsplit(":")[0]):int(roi.rsplit(",")[1].rsplit(":")[1][:-1])])
       
        data_epl = np.empty(np.shape(data_norm), dtype=float)
        cropped_view = np.linspace(start=test_matrix["Cropping Start"][jj], stop=480, num=480-test_matrix["Cropping Start"][jj]+1, dtype=int)
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
            data_epl[k, :] = cs_all[ii][j](data_norm[k, :] + offset)
            
        offset_epl = np.median(data_epl[int(roi.rsplit(",")[0].rsplit(":")[0][1:]):int(roi.rsplit(",")[0].rsplit(":")[1]), int(roi.rsplit(",")[1].rsplit(":")[0]):int(roi.rsplit(",")[1].rsplit(":")[1][:-1])])
        data_epl -= offset_epl
            
        for z, k in enumerate(cropped_view):
            warnings.filterwarnings("ignore")
            peaks, _ = find_peaks(savgol_filter(data_epl[k, :], 105, 7), width=100)
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
                             "Linear Fits Error": linfits_err[i], "R2": r2_values[i], "Offset": offset, "SNR": SNR}
            
        iteration_time = timer()
    #        if i == 0:
    #            break 
        
    #%% Calculate correction factor based on 2000 um
    axial_positions = [x["Axial Position"] for x in processed_data]
    axial_positions = [x for x in axial_positions for x in x]
    
    lateral_positions = [x["Lateral Position"] for x in processed_data]
    lateral_positions = [x for x in lateral_positions for x in x]
    
    optical_diameters = [x["Optical Diameter"] for x in processed_data]
    optical_diameters = [x for x in optical_diameters for x in x]
    
    model_epl = [x["Model EPL Diameter"] for x in processed_data]
    model_epl = [x for x in model_epl for x in x]
    
    offset = np.mean([x["Offset"] for x in processed_data])
    SNR = np.mean([x["SNR"] for x in processed_data])
    
    summary = pd.DataFrame({"Axial Position": axial_positions, "Lateral Position": lateral_positions, 
                            "Optical Diameter": optical_diameters, "Model EPL Diameter": model_epl})
    
    summary = summary.sort_values(["Axial Position", "Lateral Position"])
    summary = summary.reset_index(drop=True)
    summary_list = [v for k, v in summary.groupby("Axial Position")]
    
    positions = [np.mean(x["Axial Position"]) for x in summary_list][5:]
    optical_diameters_avg = [np.mean(x["Optical Diameter"]) for x in summary_list]
    model_epl_avg = [np.mean(x["Model EPL Diameter"]) for x in summary_list]
    cf = [x1/x2 for x1,x2 in zip(optical_diameters_avg, model_epl_avg)][5:]  
        
    f = open("R:/APS 2019-1/Imaging/" + test_name + "_corrected.pckl", "wb")
    pickle.dump([trans_corr_factor, positions, cf, cs], f)
    f.close()
        
    
    
    
    
    
    