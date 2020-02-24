# -*- coding: utf-8 -*-
"""
Calculates the correction factors based on the jet diameters.
Creates the "Analyzed" folder under /ProcessedData.

Spectra Modeling Workflow
-------------------------
tube_model -> TREX_calibration -> TREX_correctionfactor -> TREX_error -> TREX_summary

Created on Sat Sep  7 16:44:26 2019

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
mm_pix = 0.016710993

data_folder = "R:/X-ray Tomography/Spectra Model/RawData"
jets = ["smallJet", "largeJet"]
scint_plates = ["ACS", "FOS", "ALS"]
flatDarks = glob.glob(data_folder + "/flat&dark/*")
flatDarks_runs = [int(x.rsplit("Test")[1].rsplit("_")[0]) for x in flatDarks]

for jet in jets:  
    datasets = glob.glob(data_folder + "/" + jet + "/*50KI*")
    for dataset in datasets:
        run = int(dataset.rsplit("Test")[1].rsplit("_")[0])
        idx = (np.array(flatDarks_runs) - run).tolist().index(min([n for n in np.array(flatDarks_runs) - run  if n>0]))
        
        dark = np.array(Image.open("R:/X-ray Tomography/Spectra Model/ProcessedData/darkAVG/Test" + str(flatDarks_runs[idx]) + ".tif"))
        flat = np.array(Image.open("R:/X-ray Tomography/Spectra Model/ProcessedData/flatAVG/Test" + str(flatDarks_runs[idx]) + ".tif"))    
        I0 = flat - dark
        
        test_paths = glob.glob(dataset + "/*.tif")
        processed_data = len(test_paths) * [None]
        for i, data in enumerate(test_paths):
            warnings.filterwarnings("ignore")
            data_norm = (np.array(Image.open(data)) - dark) / I0
            warnings.filterwarnings("default")
            offset = 1-np.median(data_norm[400:900, 60:200])
        
            data_epl = np.empty(np.shape(data_norm), dtype=float)
            cropped_view = np.linspace(start=350, stop=950, num=950-350+1, dtype=int)
            left_bound = len(cropped_view) * [np.nan]
            right_bound = len(cropped_view) * [np.nan]
            model_epl = len(cropped_view) * [np.nan]
            optical_diameter = len(cropped_view) * [np.nan]
            axial_position = len(cropped_view) * [np.nan]
            lateral_position = len(cropped_view) * [np.nan]
            fitted_graph = len(cropped_view) * [np.nan]
            epl_graph = len(cropped_view) * [np.nan]
            actual_diameter = np.zeros((len(cropped_view),1))
            fitted_diameter = np.zeros((len(cropped_view),1))
            epl_corr_factor = np.zeros((len(cropped_view),1))
            trans_corr_factor = np.zeros((len(cropped_view),1))
            for z, k in enumerate(cropped_view):
                data_epl[k, :] = cs(data_norm[k, :] + offset)
                
            offset_epl = np.median(data_epl[400:900, 60:200])
            data_epl -= offset_epl
        
            for z, k in enumerate(cropped_view):
                warnings.filterwarnings("ignore")
                peaks, _ = find_peaks(savgol_filter(data_epl[k, :], 105, 7), width=20, prominence=0.2)
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
                    model_epl[z], fitted_graph[z], epl_graph[z] = ideal_ellipse(y=savgol_filter(data_epl[k,:], 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], full_width=full_width, relative_max=fifth_max, dx=mm_pix, units='mm')
                    warnings.filterwarnings("default")
                    optical_diameter[z] = full_width * mm_pix
                    axial_position[z] = k
                    lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))
                    actual_diameter[z] = fitted_graph[z]["a"]*2
                    fitted_diameter[z] = fitted_graph[z]["b"]
                    epl_corr_factor[z] = actual_diameter[z] / fitted_diameter[z]
                    trans_corr_factor[z] = epl_to_T50KI(actual_diameter[z]) / epl_to_T50KI(fitted_diameter[z])
                        
            if len(peaks) == 1:
                signal = np.nanmean(data_epl[400:900, int(round(np.nanmean(lateral_position)))-20:int(round(np.nanmean(lateral_position)))+20])
            else:
                signal = 0
                
            noise = np.nanstd(data_epl[400:900, 60:200])
            SNR = signal / noise
            
            processed_data[i] = {"Image": data, "Optical Diameter": optical_diameter, "Model EPL Diameter": model_epl, 
                                 "Axial Position": axial_position, "Lateral Position": lateral_position, "Left Bound": left_bound, 
                                 "Right Bound": right_bound, "SNR": SNR, "EPL CF": epl_corr_factor, "T CF": trans_corr_factor}
            
#            if i == 10:
#                break
        
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
        
        # Correction factor based on the optical/EPL diameter
        EPL_cf = [(x1)/(x2) for x1,x2 in zip(optical_diameters_avg, model_epl_avg)][10:]  

        # Correction factor based on the corresponding transmissions for the optical/EPL diameter
        T_cf = [epl_to_T50KI(x1)/epl_to_T50KI(x2) for x1,x2 in zip(optical_diameters_avg, model_epl_avg)][10:]  
        
        f = open("R:/X-ray Tomography/Spectra Model/ProcessedData/Analyzed/" + dataset.rsplit('\\')[-1] + "_analyzed.pckl", "wb")
        pickle.dump([processed_data, positions, EPL_cf, T_cf], f)
        f.close()
            
        
        
        
        
        
        





