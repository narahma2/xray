# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:14:19 2019

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

models_all = [water_model, KI1p6_model, KI3p4_model, KI4p8_model, KI8p0_model, KI10p0_model, KI11p1_model]
#%% Calculate beam attenuation coefficients [1/cm]
density = [1.0, density_KIinH2O(1.6), density_KIinH2O(3.4), density_KIinH2O(4.8), density_KIinH2O(8), 
           density_KIinH2O(10), density_KIinH2O(11.1)]
KIperc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]
spray_epl = water_model[2]

atten_water = [-np.log(np.array(x))/np.array(water_model[2])/density[0] for x in water_model[3]]
atten_KI1p6 = [-np.log(np.array(x))/np.array(KI1p6_model[2])/density[1] for x in KI1p6_model[3]]
atten_KI3p4 = [-np.log(np.array(x))/np.array(KI3p4_model[2])/density[2] for x in KI3p4_model[3]]
atten_KI4p8 = [-np.log(np.array(x))/np.array(KI4p8_model[2])/density[3] for x in KI4p8_model[3]]
atten_KI8p0 = [-np.log(np.array(x))/np.array(KI8p0_model[2])/density[4] for x in KI8p0_model[3]]
atten_KI10p0 = [-np.log(np.array(x))/np.array(KI10p0_model[2])/density[5] for x in KI10p0_model[3]]
atten_KI11p1 = [-np.log(np.array(x))/np.array(KI11p1_model[2])/density[6] for x in KI11p1_model[3]]

atten_all = [atten_water, atten_KI1p6, atten_KI3p4, atten_KI4p8, atten_KI8p0, atten_KI10p0, atten_KI11p1]
trans_all = [water_model[3], KI1p6_model[3], KI3p4_model[3], KI4p8_model[3], KI8p0_model[3], KI10p0_model[3], KI11p1_model[3]]

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
    
    fits = [[x(KIperc)[0] for x in p], [x(KIperc)[1] for x in p], [x(KIperc)[2] for x in p], [x(KIperc)[3] for x in p], 
            [x(KIperc)[4] for x in p], [x(KIperc)[5] for x in p], [x(KIperc)[6] for x in p]]

    model_fits[j] = [CubicSpline(x[::-1][:-1], spray_epl[::-1][:-1]) for x in fits]
    
#%% Find corresponding points based on a given transmission
jet2000_epl = len(atten_water) * [None]
jet2000_alpha = len(atten_water) * [None]
jet700_epl = len(atten_water) * [None]
jet700_alpha = len(atten_water) * [None]

for j, _ in enumerate(atten_water):
    # Equivalent transmission curve at 2000 um, 0% KI
    jet2000_epl[j] = np.array(spray_epl[[np.argmin(abs(trans_all[i][j] - water_model[3][j][200])) for i in range(0, len(trans_all))]])
    jet2000_alpha[j] = -np.log(water_model[3][j][200]) / density / jet2000_epl[j]
    # Equivalent transmission curve at 700 um, 11.1% KI
    jet700_epl[j] = np.array(spray_epl[[np.argmin(abs(trans_all[i][j] - KI11p1_model[3][j][70])) for i in range(0, len(trans_all))]])
    jet700_alpha[j] = -np.log(KI11p1_model[3][j][70]) / density / jet700_epl[j]
    
plt.figure()
plt.plot(KIperc, jet2000_epl[300]*10*1000, label='0% KI @ 2000 $\mu$m, T = ' + str(round(trans_all[0][300][200], 4)))
plt.plot(KIperc, jet700_epl[300]*10*1000, label='11.1% KI @ 700 $\mu$m, T = ' + str(round(trans_all[2][300][70], 4)))
plt.ylim([0, 2250])
plt.xlim([KIperc[0], KIperc[-1]])
plt.xlabel('Spray KI %')
plt.ylabel('Spray EPL [$\mu$m]')
plt.title('Equivalent Transmission Curves @ Middle')
plt.legend()
plt.grid(axis='both')

jet2000_inv = curve_fit(lambda t,a: a/t, xdata=jet2000_epl[300]*10*1000, ydata=jet2000_alpha[300])[0][0]
jet2000_inv_x = np.linspace(jet2000_epl[300][0]*10*1000, jet2000_epl[300][-1]*10*1000)
jet2000_inv_y = jet2000_inv / jet2000_inv_x
jet700_inv = curve_fit(lambda t,a: a/t, xdata=jet700_epl[300]*10*1000, ydata=jet700_alpha[300])[0][0]
jet700_inv_x = np.linspace(jet700_epl[300][0]*10*1000, jet700_epl[300][-1]*10*1000)
jet700_inv_y = jet700_inv / jet700_inv_x

plt.figure()
plt.plot(jet2000_epl[300]*10*1000, jet2000_alpha[300], linestyle='-', color='#1f77b4ff', label='0% KI @ 2000 $\mu$m')
plt.plot(jet2000_inv_x, jet2000_inv_y, linestyle='--', color='#1f77b4ff', label='0% KI @ 2000 $\mu$m Inverse Fit')
plt.plot(jet700_epl[300]*10*1000, jet700_alpha[300], linestyle='-', color='#ff7f0eff', label='11.1% KI @ 700 $\mu$m')
plt.plot(jet700_inv_x, jet700_inv_y, linestyle='--', color='#ff7f0eff', label='11.1% KI @ 700 $\mu$m Inverse Fit')
plt.xlim([0, 2250])
plt.xlabel('Spray EPL [$\mu$m]')
plt.ylabel('Attenuation Coefficient [1/cm]')
plt.title('Equivalent Transmission Curves @ Middle')
plt.legend()
plt.grid(axis='both')

#%% Fitting of attenuation coefficient against the KI% for a given EPL
plt.figure()
plt.scatter(KIperc, [y[300][200] for y in atten_all], marker='o', color='#1f77b4ff', label='2000 $\mu$m Data')
plt.plot(KIperc, pfits[300][200](KIperc), linestyle='--', color='#1f77b4ff', label='2000 $\mu$m Fit')
plt.scatter(KIperc, [y[300][70] for y in atten_all], marker='v', color='#ff7f0eff', label='700 $\mu$m Data')
plt.plot(KIperc, pfits[300][70](KIperc), linestyle='--', color='#ff7f0eff', label='700 $\mu$m Fit')
plt.xlabel('KI %')
plt.ylabel('Attenuation Coefficient [1/cm]')
plt.title('Attenuation Coefficient vs. KI %')
plt.legend()
plt.grid(axis='both')
plt.xlim([-1, 11.5])

#%% Fitting of transmission against the KI% for a given EPL
plt.figure()
plt.plot(KIperc, pTfits[300][200](KIperc), label='2000 $\mu$m Curve')
plt.plot(KIperc, pTfits[300][70](KIperc), label='700 $\mu$m Curve')
plt.xlabel('KI %')
plt.ylabel('Transmission [I/I$_0$]')
plt.title('Transmission vs. KI %')
plt.legend()
plt.grid(axis='both')
plt.xlim([0, 11.1])
plt.ylim([0, 1])

#%% Plots
plt.figure()
plt.plot(spray_epl, atten_water[300], label='0%')
plt.plot(spray_epl, atten_KI1p6[300], label='1.6%')
plt.plot(spray_epl, atten_KI3p4[300], label='3.4%')
plt.plot(spray_epl, atten_KI4p8[300], label='4.8%')
plt.plot(spray_epl, atten_KI8p0[300], label='8.0%')
plt.plot(spray_epl, atten_KI10p0[300], label='10.0%')
plt.plot(spray_epl, atten_KI11p1[300], label='11.1%')
plt.legend()
plt.xlabel('Spray EPL [cm]')
plt.ylabel('Attenuation Coefficient [1/cm]')
plt.title('Attenuation Coefficients @ Middle')   

plt.figure()
plt.plot(spray_epl, water_model[3][300], label='0%')
plt.plot(spray_epl, KI1p6_model[3][300], label='1.6%')
plt.plot(spray_epl, KI3p4_model[3][300], label='3.4%')
plt.plot(spray_epl, KI4p8_model[3][300], label='4.8%')
plt.plot(spray_epl, KI8p0_model[3][300], label='8.0%')
plt.plot(spray_epl, KI10p0_model[3][300], label='10.0%')
plt.plot(spray_epl, KI11p1_model[3][300], label='11.1%')
plt.legend()
plt.xlabel('Spray EPL [cm]')
plt.ylabel('Transmission [I/I$_0$]')
plt.title('Transmission Curves @ Middle')   
    
    
#%% Attenuation coefficient correction for 2000um, 0% KI case
cm_pix = 700/84 / (1000*10)   # See "APS White Beam.xlsx -> Spatial Resolution"
dark = np.array(Image.open("R:/APS 2018-1/Imaging/Processed/Jets/AVG_Jet_dark2.tif"))
flatfieldavg = np.array(Image.open("R:/APS 2018-1/Imaging/Processed/Jets/AVG_Jet_flat2.tif"))
flatfieldavg_darksub = flatfieldavg - dark
beam_middle = np.zeros((flatfieldavg_darksub.shape[1]))
for i in range(flatfieldavg_darksub.shape[1]):
    warnings.filterwarnings("ignore")
    beam_middle[i] = np.argmax(savgol_filter(flatfieldavg_darksub[:,i], 55, 3))
    warnings.filterwarnings("default")

beam_middle_avg = int(np.mean(beam_middle).round())

angles = water_model[0]
angle_to_px = [beam_middle_avg+3500*np.tan(x*10**-3)/cm_pix for x in angles]

iii = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
jjj = [14, 15, 16, 17, 18, 19, 43, 0, 1, 2, 3, 4, 5, 6]

for nn, _ in enumerate(iii):
    ii = iii[nn]
    jj = jjj[nn]
    if jj > 6:
        prom = 0.08
    else:
        prom = 0.04
    
    test_matrix = pd.read_csv("R:/APS 2018-1/Imaging/APS White Beam.txt", sep="\t+", engine="python")
    test_name = test_matrix["Test"][jj]
    test_path = "R:\\APS 2018-1\\Imaging\\Jets Big and Day\\" + test_name
    tests = glob.glob(test_path + "\\*.tif")
    
    actual_diameter = np.zeros((len(tests),1))
    fitted_diameter = np.zeros((len(tests),1))
    cropped_view = np.linspace(start=1, stop=340, num=340, dtype=int)
    for i, tt in enumerate(tests):        
        data = np.array(Image.open(tt))
        warnings.filterwarnings("ignore")
        data_norm = (data-dark) / flatfieldavg_darksub
        warnings.filterwarnings("default")
        offset = 1-np.median(data_norm[50:300, 25:125])
           
        j = np.argmin(abs(beam_middle_avg-np.array(angle_to_px)))
        data_epl = models_all[ii][1][j](data_norm[beam_middle_avg, :] + offset)
    
        peaks, _ = find_peaks(savgol_filter(data_epl, 105, 7), width=50, prominence=prom)
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
    cs = [CubicSpline(np.array(models_all[ii][3][n][::-2]) / np.mean(trans_corr_factor), spray_epl[::-2]) for n,_ in enumerate(models_all[ii][3])]
    
    corrected_water_model = [spray_epl, cs, models_all[ii][3], [cs[n](models_all[ii][3][n]) for n,_ in enumerate(models_all[ii][3])]] 
        
    #%% Applying the fixed transmission
    actual_diameter_cf = np.zeros((len(tests),1))
    fitted_diameter_cf = np.zeros((len(tests),1))
    cropped_view = np.linspace(start=1, stop=340, num=340, dtype=int)
    for i, tt in enumerate(tests):        
        data = np.array(Image.open(tt))
        warnings.filterwarnings("ignore")
        data_norm = (data-dark) / flatfieldavg_darksub
        warnings.filterwarnings("default")
        offset = 1-np.median(data_norm[50:300, 25:125])
           
        j = np.argmin(abs(beam_middle_avg-np.array(angle_to_px)))
        data_epl = corrected_water_model[1][j](data_norm[beam_middle_avg, :] + offset)
    
        peaks, _ = find_peaks(savgol_filter(data_epl, 105, 7), width=50, prominence=prom)
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
        data_norm = (data-dark) / flatfieldavg_darksub
        warnings.filterwarnings("default")
        offset = 1-np.median(data_norm[50:300, 25:125])
        
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
            data_epl[k, :] = corrected_water_model[1][j](data_norm[k, :] + offset)
            
        for z, k in enumerate(cropped_view):
            warnings.filterwarnings("ignore")
            peaks, _ = find_peaks(savgol_filter(data_epl[k, :], 105, 7), width=20, prominence=prom)
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
        
    f = open("R:/APS 2018-1/Imaging/Processed/Corrected_Aug18/" + test_name + "_corrected.pckl", "wb")
    pickle.dump([trans_corr_factor, positions, cf, cs], f)
    f.close()
    
#%% Load in 700um case    
    
    
    
    
    