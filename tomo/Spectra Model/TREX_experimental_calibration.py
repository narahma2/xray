# -*- coding: utf-8 -*-
"""
Calculates the experimental correction factors based on the jet diameters.

Experimental CF will be used directly in the image processing for the 2018 tomo data sets.
Created on Mon March  2 09:32:00 2020

@author: rahmann
"""

import sys
if sys.platform == 'win32':
    sys.path.append('E:/GitHub/xray/general')
    sys.path.append('E:/GitHub/xray/temperature')
    sys_folder = 'R:'
elif sys.platform == 'linux':
    sys.path.append('/mnt/e/GitHub/xray/general')
    sys.path.append('/mnt/e/GitHub/xray/temperature')
    sys_folder = '/mnt/r/'

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
        
            cropped_view = np.linspace(start=350, stop=950, num=950-350+1, dtype=int)
            left_bound = len(cropped_view) * [np.nan]
            right_bound = len(cropped_view) * [np.nan]
            optical_diameter = len(cropped_view) * [np.nan]
            axial_position = len(cropped_view) * [np.nan]
            lateral_position = len(cropped_view) * [np.nan]
            norm_diameter = len(cropped_view) * [np.nan]
            fitted_graph = len(cropped_view) * [np.nan]
            norm_graph = len(cropped_view) * [np.nan]
            actual_diameter = np.zeros((len(cropped_view),1))
            norm_correction = np.zeros((len(cropped_view),1))
            
            for z, k in enumerate(cropped_view):
                warnings.filterwarnings("ignore")
                peaks, _ = find_peaks(savgol_filter(data_norm[k, :], 105, 7), width=20, prominence=0.2)
                warnings.filterwarnings("default")
                
                if len(peaks) == 1:
                    warnings.filterwarnings("ignore")
                    [fifth_width, fifth_max, lpos, rpos] = peak_widths(savgol_filter(data_norm[k, :], 105, 7), peaks, rel_height=0.8)
                    warnings.filterwarnings("default")
                    fifth_width = fifth_width[0]
                    fifth_max = fifth_max[0]
                    left_bound[z] = lpos[0]
                    right_bound[z] = rpos[0]
                    warnings.filterwarnings("ignore")
                    norm_diameter[z], fitted_graph[z], data_graph[z] = ideal_ellipse(y=savgol_filter(data_norm[k,:], 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], relative_width=fifth_width, relative_max=fifth_max, dx=mm_pix, units='mm')
                    warnings.filterwarnings("default")
                    optical_diameter[z] = fifth_width * mm_pix
                    axial_position[z] = k
                    lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))
                    actual_diameter[z] = fitted_graph[z]["a"]*2
                    norm_correction[z] = actual_diameter[z] / norm_diameter[z]
                        
            if len(peaks) == 1:
                signal = np.nanmean(data_norm[400:900, int(round(np.nanmean(lateral_position)))-20:int(round(np.nanmean(lateral_position)))+20])
            else:
                signal = 0
                
            noise = np.nanstd(data_norm[400:900, 60:200])
            SNR = signal / noise
            
            processed_data[i] = {"Image": data, "Optical Diameter": optical_diameter, "X-ray Diameter": norm_diameter, 
                                 "Axial Position": axial_position, "Lateral Position": lateral_position, "Left Bound": left_bound, 
                                 "Right Bound": right_bound, "SNR": SNR, "T CF": norm_correction}
            
#            if i == 10:
#                break
        
        # Calculate correction factor
        axial_positions = [x["Axial Position"] for x in processed_data]
        axial_positions = [x for x in axial_positions for x in x]
        
        lateral_positions = [x["Lateral Position"] for x in processed_data]
        lateral_positions = [x for x in lateral_positions for x in x]
        
        optical_diameters = [x["Optical Diameter"] for x in processed_data]
        optical_diameters = [x for x in optical_diameters for x in x]
        
        xray_diameter = [x["X-ray Diameter"] for x in processed_data]
        xray_diameter = [x for x in xray_diameter for x in x]
        
        SNR = np.mean([x["SNR"] for x in processed_data])
        
        summary = pd.DataFrame({"Axial Position": axial_positions, "Lateral Position": lateral_positions, 
                                "Optical Diameter": optical_diameters, "X-ray Diameter": xray_diameter})
        
        summary = summary.sort_values(["Axial Position", "Lateral Position"])
        summary = summary.reset_index(drop=True)
        summary_list = [v for k, v in summary.groupby("Axial Position")]
        
        positions = [np.mean(x["Axial Position"]) for x in summary_list][10:]
        optical_diameters_avg = [np.mean(x["Optical Diameter"]) for x in summary_list]
        xray_diameter_avg = [np.mean(x["X-ray Diameter"]) for x in summary_list]  
        
        # Correction factor based on the optical/X-ray diameter
        experimental_cf = [(x1)/(x2) for x1,x2 in zip(optical_diameters_avg, model_epl_avg)][10:]  
        # Save processed_data
        f = open("R:/X-ray Tomography/Spectra Model/ProcessedData/Experimental/" + dataset.rsplit('\\')[-1] + "_experimental.pckl", "wb")
        pickle.dump([processed_data, positions, experimental_cf], f)
        f.close()
        
        # Save summary of results
        f = open("R:/X-ray Tomography/Spectra Model/ProcessedData/Experimental/" + dataset.rsplit('\\')[-1] + "_summary.pckl", "wb")
        pickle.dump([summary_list, experimental_cf], f)
        f.close()