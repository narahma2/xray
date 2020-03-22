# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:22:05 2019

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
from Spectra.spectrum_modeling import multi_angle as xop, xcom, xcom_reshape, density_KIinH2O, beer_lambert, visible_light, beer_lambert_unknown
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, find_peaks, peak_widths
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

#%% Match transmissions based on 2000um, pure water case
t_water_2000 = [x[200] for x in water_model[3]]

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

p = len(spray_epl) * [None]
y_err = len(spray_epl) * [None]
r2 = len(spray_epl) * [None]
pfits = len(atten_water) * [None]
model_fits = len(atten_water) * [None]  
    
for j, _ in enumerate(atten_water):
    for i, _ in enumerate(spray_epl):
        # Fitting of attenuation coefficient of each model against the KI % of that model
        # Each "i" represents each of the individual EPL values tested
        p[i], y_err[i], r2[i] = lin_fit(KIperc, [y[j][i] for y in atten_all])
        
    # Each "j" is the vertical position in the WB. "i" is the individual EPL.
    # pfits[300][70](1.6) : Returns the attenuation coefficient at middle of WB, for EPL = 700 um and KI% = 1.6%
    pfits[j] = [lin_fit(KIperc, [x[j][k] for x in atten_all])[0] for k,_ in enumerate(spray_epl)]
        
    fits = [[x(KIperc)[0] for x in p], [x(KIperc)[1] for x in p], [x(KIperc)[2] for x in p], [x(KIperc)[3] for x in p], 
            [x(KIperc)[4] for x in p], [x(KIperc)[5] for x in p], [x(KIperc)[6] for x in p]]

    model_fits[j] = [CubicSpline(x[::-1][:-1], spray_epl[::-1][:-1]) for x in fits]

f = open("R:/APS 2018-1/Imaging/interpolation_model.pckl", "wb")
pickle.dump([spray_epl, KIperc, model_fits], f)
f.close()

plt.figure()
plt.plot(water_model[3][300], spray_epl, label='Water')
plt.plot(KI1p6_model[3][300], spray_epl, label='1.6%')
plt.plot(KI3p4_model[3][300], spray_epl, label='3.4%')
plt.plot(KI4p8_model[3][300], spray_epl, label='4.8%')
plt.plot(KI8p0_model[3][300], spray_epl, label='8.0%')
plt.plot(KI10p0_model[3][300], spray_epl, label='10.0%')
plt.plot(KI11p1_model[3][300], spray_epl, label='11.1%')
plt.xlabel('Transmission')
plt.ylabel('Spray EPL')
plt.legend()

#%%
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

test_matrix = pd.read_csv("R:/APS 2018-1/Imaging/APS White Beam.txt", sep="\t+", engine="python")
test_name = test_matrix["Test"][0]
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
    data_epl = water_model[1][j](data_norm[beam_middle_avg, :] + offset)

    peaks, _ = find_peaks(savgol_filter(data_epl, 105, 7), width=20, prominence=0.1)
    [full_width, fifth_max, lpos, rpos] = peak_widths(savgol_filter(data_epl, 105, 7), peaks, rel_height=0.8)
    full_width = full_width[0]
    fifth_max = fifth_max[0]
    left_bound = lpos[0]
    right_bound = rpos[0]

    model_epl, fitted_graph, epl_graph = ideal_ellipse(y=savgol_filter(data_epl, 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], 
                                                       full_width=full_width, relative_max=fifth_max, dx=cm_pix)
    
    actual_diameter[i] = fitted_graph["a"]*2
    fitted_diameter[i] = fitted_graph["b"]

actual_epl = np.argmin(abs(spray_epl-np.mean(actual_diameter)))
fitted_epl = np.argmin(abs(spray_epl-np.mean(fitted_diameter)))
atten_water_cf = atten_water[300][actual_epl] / atten_water[300][fitted_epl]
trans_water_cf = water_model[3][300][actual_epl] / water_model[3][300][fitted_epl]

# Cubic spline fitting of Transmission vs. spray_epl curves (needs to be reversed b/c of monotonically increasing
# restriction on 'x', however this does not change the interpolation call)
cs = [CubicSpline(np.array(water_model[3][n][::-2]), np.array(atten_water[n][::-2])*atten_water_cf) for n,_ in enumerate(water_model[3])]

corrected_water_model = [spray_epl, cs, water_model[3], [cs[n](water_model[3][n]) for n,_ in enumerate(water_model[3])]]

#%% Fixed images
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
    data_alpha = cs[j](data_norm[beam_middle_avg, :] + offset)
#    data_epl = -np.log(np.array(data_norm[beam_middle_avg, :])) / density[0] / np.array(data_alpha)
    data_epl = spray_epl[[np.nanargmin(abs(atten_water[j]-x)) for x in data_alpha]]
#    data_epl = spray_epl[[np.nanargmin(abs(corrected_water_model[3][j]-x)) for x in data_alpha]]

    peaks, _ = find_peaks(savgol_filter(data_epl, 105, 7), width=20, prominence=0.1)
    [full_width, fifth_max, lpos, rpos] = peak_widths(savgol_filter(data_epl, 105, 7), peaks, rel_height=0.8)
    full_width = full_width[0]
    fifth_max = fifth_max[0]
    left_bound = lpos[0]
    right_bound = rpos[0]

    model_epl, fitted_graph, epl_graph = ideal_ellipse(y=savgol_filter(data_epl, 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], 
                                                       full_width=full_width, relative_max=fifth_max, dx=cm_pix)
    
    actual_diameter_cf[i] = fitted_graph["a"]*2
    fitted_diameter_cf[i] = fitted_graph["b"]

#%%
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
    
    data_alpha = np.empty(np.shape(data_norm), dtype=float)
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
        data_alpha[k, :] = corrected_water_model[1][j](data_norm[k, :] + offset)
        data_epl[k, :] = spray_epl[[np.nanargmin(abs(atten_water[j]-x)) for x in data_alpha[k, :]]]
        
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
    if np.mod(i, 50) == 0:
        print(str(round(100*(i+1)/len(tests))) + "% completed. " + 
              str(round((iteration_time - loop_start)/60, 2)) + " minutes in. ETA: " + 
              str(round(((iteration_time - loop_start)/((i+1)/len(tests)) - (iteration_time - loop_start))/60, 2)) + " minutes left.")
        
#    if i == 100:
#        break
        
f = open("R:/APS 2018-1/Imaging/Processed/Corrected/" + test_name + "_corrected.pckl", "wb")
pickle.dump(processed_data, f)
f.close()

processed_data = processed_data[:i+1]

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

#%% Apply results to a 700um, 0% KI case
case = 15
test_name = test_matrix["Test"][case]
test_KI = test_matrix["KI %"][case]
test_path = "R:\\APS 2018-1\\Imaging\\Jets Big and Day\\" + test_name
tests = glob.glob(test_path + "\\*.tif")

processed_data700 = len(tests) * [None]

loop_start = timer()
for i, tt in enumerate(tests):        
    data700 = np.array(Image.open(tt))
    warnings.filterwarnings("ignore")
    data_norm700 = (data700-dark) / flatfieldavg_darksub
    warnings.filterwarnings("default")
    offset = 1-np.median(data_norm700[50:300, 25:125])
    
    data_alpha700 = np.empty(np.shape(data_norm700), dtype=float)
    data_epl700 = np.empty(np.shape(data_norm700), dtype=float)
    cropped_view = np.linspace(start=1, stop=340, num=340, dtype=int)
    left_bound = len(cropped_view) * [np.nan]
    right_bound = len(cropped_view) * [np.nan]
    model_epl700 = len(cropped_view) * [np.nan]
    optical_diameter700 = len(cropped_view) * [np.nan]
    axial_position = len(cropped_view) * [np.nan]
    lateral_position = len(cropped_view) * [np.nan]
    fitted_graph700 = len(cropped_view) * [np.nan]
    epl_graph700 = len(cropped_view) * [np.nan]
    
    for z, k in enumerate(cropped_view):
        j = np.argmin(abs(k-np.array(angle_to_px)))
        data_alpha700[k, :] = corrected_water_model[1][j](data_norm700[k, :] + offset)
#        epl700 = spray_epl[[np.nanargmin(abs(atten_water[j]-x)) for x in data_alpha700[k, :]]]
#        epl_ind = [np.nanargmin(abs(x-spray_epl)) for x in epl700]
#        alpha_KI_corr = data_alpha700[k, :] - [pfits[j][i].c[0]*(test_KI-0) for i in epl_ind]
#        data_alpha700[k, :] = alpha_KI_corr
        data_epl700[k, :] = spray_epl[[np.nanargmin(abs(atten_water[j]-x)) for x in data_alpha700[k, :]]]
        
    for z, k in enumerate(positions):
        data_epl700[int(k), :] = data_epl700[int(k), :] * cf[int(z)]
        
    for z, k in enumerate(cropped_view):
        warnings.filterwarnings("ignore")
        peaks, _ = find_peaks(savgol_filter(data_epl700[k, :], 105, 7), width=20, prominence=0.04)
        warnings.filterwarnings("default")
        
        if len(peaks) == 1:
            warnings.filterwarnings("ignore")
            [full_width, fifth_max, lpos, rpos] = peak_widths(savgol_filter(data_epl700[k, :], 105, 7), peaks, rel_height=0.8)
            warnings.filterwarnings("default")
            full_width = full_width[0]
            fifth_max = fifth_max[0]
            left_bound[z] = lpos[0]
            right_bound[z] = rpos[0]
            warnings.filterwarnings("ignore")
            model_epl700[z], fitted_graph700[z], epl_graph700[z] = ideal_ellipse(y=savgol_filter(data_epl700[k,:], 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], full_width=full_width, relative_max=fifth_max, dx=cm_pix)
            warnings.filterwarnings("default")
            optical_diameter700[z] = full_width * cm_pix
            axial_position[z] = k
            lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))
    
    if len(peaks) == 1:
        signal = np.nanmean(data_epl700[10:340, int(round(np.nanmean(lateral_position)))-20:int(round(np.nanmean(lateral_position)))+20])
    else:
        signal = 0
        
    noise = np.nanstd(data_epl700[50:300, 25:125])
    SNR = signal / noise
    
    processed_data700[i] = {"Image": tt, "Optical Diameter": optical_diameter700, "Model EPL Diameter": model_epl700, "Axial Position": axial_position,
                         "Lateral Position": lateral_position, "Left Bound": left_bound, "Right Bound": right_bound, 
                         "Offset": offset, "SNR": SNR}
        
    iteration_time = timer()
    if np.mod(i, 50) == 0:
        print(str(round(100*(i+1)/len(tests))) + "% completed. " + 
              str(round((iteration_time - loop_start)/60, 2)) + " minutes in. ETA: " + 
              str(round(((iteration_time - loop_start)/((i+1)/len(tests)) - (iteration_time - loop_start))/60, 2)) + " minutes left.")
        
#    if i == 0:
#        break
        
f = open("R:/APS 2018-1/Imaging/Processed/Corrected/" + test_name + "_corrected.pckl", "wb")
pickle.dump(processed_data700, f)
f.close()

processed_data700 = processed_data700[:i+1]

#%% CF calculation for the 700um case after 2000um calibration
axial_positions700 = [x["Axial Position"] for x in processed_data700]
axial_positions700 = [x for x in axial_positions700 for x in x]

lateral_positions700 = [x["Lateral Position"] for x in processed_data700]
lateral_positions700 = [x for x in lateral_positions700 for x in x]

optical_diameters700 = [x["Optical Diameter"] for x in processed_data700]
optical_diameters700 = [x for x in optical_diameters700 for x in x]

model_epl700 = [x["Model EPL Diameter"] for x in processed_data700]
model_epl700 = [x for x in model_epl700 for x in x]

offset = np.mean([x["Offset"] for x in processed_data700])
SNR = np.mean([x["SNR"] for x in processed_data700])

summary700 = pd.DataFrame({"Axial Position": axial_positions700, "Lateral Position": lateral_positions700, 
                        "Optical Diameter": optical_diameters700, "Model EPL Diameter": model_epl700})

summary700 = summary700.sort_values(["Axial Position", "Lateral Position"])
summary700 = summary700.reset_index(drop=True)
summary700_list = [v for k, v in summary700.groupby("Axial Position")]

positions700 = [np.mean(x["Axial Position"]) for x in summary700_list][5:]
optical_diameters700_avg = [np.mean(x["Optical Diameter"]) for x in summary700_list]
model_epl700_avg = [np.mean(x["Model EPL Diameter"]) for x in summary700_list]
cf700 = [x1/x2 for x1,x2 in zip(optical_diameters700_avg, model_epl700_avg)][5:]





