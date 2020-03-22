# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:55:14 2019

@author: rahmann
"""

import sys
sys.path.append('E:/OneDrive - purdue.edu/Research/GitHub/coding/python')

import pickle
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, find_peaks, peak_widths
from skimage.transform import rotate
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse

def process_images(test_name, model, master_positions, master_cf, prom, rot):
    test_path = "R:\\APS 2018-1\\Imaging\\Jets Big and Day\\" + test_name
    tests = glob.glob(test_path + "\\*.tif")
    processed_data = len(tests) * [None]
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
            data_epl[k, :] = model[j](data_norm[k, :] + offset)
            
        for z, k in enumerate(master_positions):
            data_epl[int(k), :] *= master_cf[z]
            
        offset_epl = np.median(data_epl[50:300, 25:125])
        data_epl -= offset_epl
        data_epl = rotate(data_epl, rot)    
        
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
        
        processed_data[i] = {"Image": tt, "Optical Diameter": optical_diameter, "Model EPL Diameter": model_epl, 
                             "Axial Position": axial_position, "Lateral Position": lateral_position, "Left Bound": left_bound, 
                             "Right Bound": right_bound, "SNR": SNR}
    
    # Calculate correction factor
    axial_positions = [x["Axial Position"] for x in processed_data]
    axial_positions = [x for x in axial_positions for x in x]
    
    lateral_positions = [x["Lateral Position"] for x in processed_data]
    lateral_positions = [x for x in lateral_positions for x in x]
    
    optical_diameters = [x["Optical Diameter"] for x in processed_data]
    optical_diameters = [x for x in optical_diameters for x in x]
    
    model_epl = [x["Model EPL Diameter"] for x in processed_data]
    model_epl = [x for x in model_epl for x in x]
    
    SNR = np.mean([x["SNR"] for x in processed_data])
    
    summary = pd.DataFrame({"Axial Position": axial_positions, "Lateral Position": lateral_positions, 
                            "Optical Diameter": optical_diameters, "Model EPL Diameter": model_epl})
    
    summary = summary.sort_values(["Axial Position", "Lateral Position"])
    summary = summary.reset_index(drop=True)
    summary_list = [v for k, v in summary.groupby("Axial Position")]
    
    positions = [np.mean(x["Axial Position"]) for x in summary_list][10:]
    optical_diameters_avg = [np.mean(x["Optical Diameter"]) for x in summary_list]
    model_epl_avg = [np.mean(x["Model EPL Diameter"]) for x in summary_list]
    cf = [x1/x2 for x1,x2 in zip(optical_diameters_avg, model_epl_avg)][10:]  
    
    f = open("R:/APS 2018-1/Imaging/Processed/Analyzed_Aug23/" + test_name + "_analyzed.pckl", "wb")
    pickle.dump([processed_data, positions, cf], f)
    f.close()

#%% Initial setup
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

spray_epl = water_model[2][1:]
water_model[3] = [water_model[3][i][1:] for i,_ in enumerate(water_model[3])]
KI1p6_model[3] = [KI1p6_model[3][i][1:] for i,_ in enumerate(KI1p6_model[3])]
KI3p4_model[3] = [KI3p4_model[3][i][1:] for i,_ in enumerate(KI3p4_model[3])]
KI4p8_model[3] = [KI4p8_model[3][i][1:] for i,_ in enumerate(KI4p8_model[3])]
KI8p0_model[3] = [KI8p0_model[3][i][1:] for i,_ in enumerate(KI8p0_model[3])]
KI10p0_model[3] = [KI10p0_model[3][i][1:] for i,_ in enumerate(KI10p0_model[3])]
KI11p1_model[3] = [KI11p1_model[3][i][1:] for i,_ in enumerate(KI11p1_model[3])]
trans_all = [water_model[3], KI1p6_model[3], KI3p4_model[3], KI4p8_model[3], KI8p0_model[3], KI10p0_model[3], KI11p1_model[3]]

test_matrix = pd.read_csv("R:/APS 2018-1/Imaging/APS White Beam.txt", sep="\t+", engine="python")

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

#%% 700um cases
corrected_folder = "R:\\APS 2018-1\\Imaging\\Processed\\Corrected_Aug18"
tests = glob.glob(corrected_folder + "\\*700*.pckl")

KIconc = [0.0, 1.6, 3.4, 4.8, 8.0, 10.0, 11.1]
trcf_700 = []
test_ind_2000 = [0, 1, 2, 3, 4, 5, 6]

plt.figure()
for i, test in enumerate(tests):
    f = open(test, "rb")
    trans_corr_factor, positions, cf, cs = pickle.load(f)
    f.close()
    
    trcf_700.append(np.mean(trans_corr_factor))
    plt.plot(savgol_filter(trans_corr_factor, 55, 3), label=str(KIconc[i]) + '% KI')
plt.title('Transmission Correction Factor for 700 $\mu$m')
plt.ylabel('Middle Trans. CF [-]')
plt.xlabel('Image')
plt.legend()
plt.ylim([1.0, 1.05])
    
#for i, test in enumerate(tests):
#    test_name = test_matrix["Test"][test_ind_2000[i]]
#    
#    cs = [CubicSpline(trans_all[i][n][::-1] / np.mean(trcf_700), spray_epl[::-1]) for n,_ in enumerate(trans_all[i])]
#    
#    process_images(test_name=test_name, model=cs, master_positions=positions, master_cf=cf, prom=0.04, rot=0)

#%% 2000um cases
corrected_folder = "R:\\APS 2018-1\\Imaging\\Processed\\Corrected_Aug18"
tests = glob.glob(corrected_folder + "\\*2000*.pckl")

KIconc = [0.0, 1.6, 3.4, 4.8, 8.0, 10.0, 11.1]
trcf_2000 = []
test_ind_700 = [14, 15, 16, 17, 18, 19, 43]

plt.figure()
for i, test in enumerate(tests):
    f = open(test, "rb")
    trans_corr_factor, positions, cf, _ = pickle.load(f)
    f.close()
    
    trcf_2000.append(np.mean(trans_corr_factor))
    plt.plot(savgol_filter(trans_corr_factor, 55, 3), label=str(KIconc[i]) + '% KI')
plt.title('Transmission Correction Factor for 2000 $\mu$m')
plt.ylabel('Middle Trans. CF [-]')
plt.xlabel('Image')
plt.legend()
plt.ylim([1.0, 1.20])
    
#for i, test in enumerate(tests):
#    test_name = test_matrix["Test"][test_ind_700[i]]
#    
#    cs = [CubicSpline(trans_all[i][n][::-1] / np.mean(trcf_2000), spray_epl[::-1]) for n,_ in enumerate(trans_all[i])]
#    
#    process_images(test_name=test_name, model=cs, master_positions=positions, master_cf=cf, prom=0.1, rot=2.0)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    