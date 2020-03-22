# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:48:01 2019

@author: rahmann
"""

import sys
sys.path.append('E:/OneDrive - purdue.edu/Research/GitHub/coding/python')

import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from Statistics.CIs_LinearRegression import lin_fit

# Functions
def ellipse(test, position):
    cm_pix = 20 / 2.1 / (1000*10) 
    f = open("R:/APS 2019-1/KI4p8_model.pckl", "rb")
    KI4p8_model = pickle.load(f)
    f.close()
    
    dark = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_dark_current.tif'))
    flatfield = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_Background_NewYAG.tif'))
    flatfield_darksub = flatfield - dark

    beam_middle = np.zeros((flatfield_darksub.shape[1]))
    for i in range(flatfield_darksub.shape[1]):
        beam_middle[i] = np.argmax(savgol_filter(flatfield_darksub[:,i], 55, 3))

    beam_middle_avg = int(np.mean(beam_middle).round())
    
    angles = KI4p8_model[0]
    angle_to_px = [beam_middle_avg+3500*np.tan(x*10**-3)/cm_pix for x in angles]

    data = np.array(Image.open("R:\\APS 2019-1\\Imaging\\Raw_WB_Images\\" + test))
    data_norm = (data-dark) / flatfield_darksub
    
    j = np.argmin(abs(position-np.array(angle_to_px)))
    data_epl = KI4p8_model[1][j](data_norm[position, :])
    
    peaks, _ = find_peaks(savgol_filter(data_epl, 105, 7), width=100)
    [fwhm, half_max, lpos, rpos] = peak_widths(savgol_filter(data_epl, 105, 7), peaks, rel_height=0.5)
    fwhm = fwhm[0]
    half_max = half_max[0]
    model_epl, fitted_graph, epl_graph = ideal_ellipse(y=savgol_filter(data_epl, 105, 7)[int(round(lpos[0])):int(round(rpos[0]))], fwhm=fwhm, half_max=half_max, dx=cm_pix)
    plot_ellipse(epl_graph, fitted_graph)
    
def ideal_ellipse(y, fwhm, half_max, dx):
    x = np.linspace(start=-(len(y)*dx)/2, stop=(len(y)*dx)/2, num=len(y))
    y = y - half_max
    
    epl_area = np.trapz(y, dx=dx)
    b = np.linspace(0,1,1000)
    a = fwhm/2*dx
    minimize = np.zeros(len(b))
    
    for i, R in enumerate(b):
        check_area = (1/2)*np.pi*a*R
        minimize[i] = abs(epl_area - check_area)
        
    y = y + half_max
        
    fitted_diameter = b[np.argmin(minimize)]
    epl_graph = {"x": x, "y": y}
    fitted_graph = {"center": (0, half_max), "a": a, "b": fitted_diameter}
    
    return fitted_diameter, fitted_graph, epl_graph

def plot_ellipse(epl_graph, fitted_graph):
    t = np.linspace(0, np.pi)
    a = fitted_graph["a"]
    b = fitted_graph["b"]
    xc = fitted_graph["center"][0]
    yc = fitted_graph["center"][1]
    
    plt.figure()
    plt.plot(epl_graph["x"], epl_graph["y"], label="EPL")
    plt.plot(xc+a*np.cos(t), yc+b*np.sin(t), label="Fitted Ellipse")
    plt.legend()

#%%
tests = glob.glob("R:/APS 2019-1/Imaging/Processed_WB_Images/KI/Calibration/*.pckl", recursive=True)
calibration_dataset = len(tests) * [None]
cf = len(tests) * [None]
positions = len(tests) * [None]
lateral_position = []

#%%
for i, test in enumerate(tests):
    f = open(test, "rb")
    processed_data = pickle.load(f)
    f.close()
    processed_data = processed_data[0:50]
    
    calibration_dataset[i] = len(processed_data) * [None]
    
    for j, _ in enumerate(processed_data):
        calibration_dataset[i][j] = {"Image": processed_data[j]["Image"].rsplit("Raw_WB_Images\\")[1],
                                     "Left Bound": processed_data[j]["Left Bound"],
                                     "Right Bound": processed_data[j]["Right Bound"],
                                     "Axial Positions":  processed_data[j]["Axial Position"],
                                     "Lateral Positions": processed_data[j]["Lateral Position"], 
                                     "Optical Diameters": processed_data[j]["Optical Diameter"],
                                     "Model EPL Diameters": processed_data[j]["Model EPL Diameter"],
                                     "C1": processed_data[j]["Linear Fits"].c[0],
                                     "C2": processed_data[j]["Linear Fits"].c[1]}
            
    axial_positions = [x["Axial Positions"] for x in calibration_dataset[i]]
    axial_positions = [x for x in axial_positions for x in x]
    
    lateral_positions = [x["Lateral Positions"] for x in calibration_dataset[i]]
    lateral_positions = [x for x in lateral_positions for x in x]
    lateral_position.append(np.nanmean(lateral_positions))
    
    optical_diameters = [x["Optical Diameters"] for x in calibration_dataset[i]]
    optical_diameters = [x for x in optical_diameters for x in x]
    
    model_epl = [x["Model EPL Diameters"] for x in calibration_dataset[i]]
    model_epl = [x for x in model_epl for x in x]
    
    summary = pd.DataFrame({"Axial Position": axial_positions, "Lateral Position": lateral_positions, 
                            "Optical Diameter": optical_diameters, "Model EPL Diameter": model_epl})
    
    summary = summary.sort_values(["Axial Position", "Lateral Position"])
    summary = summary.reset_index(drop=True)
    summary_list = [v for k, v in summary.groupby("Axial Position")]
    
    positions[i] = [np.mean(x["Axial Position"]) for x in summary_list]
    optical_diameters_avg = [np.mean(x["Optical Diameter"]) for x in summary_list]
    model_epl_avg = [np.mean(x["Model EPL Diameter"]) for x in summary_list]
    cf[i] = [x1/x2 for x1,x2 in zip(optical_diameters_avg, model_epl_avg)]
    
    summary = None
    summary_list = None

cf_mean = [np.mean(x) for x in cf]
cf_std = [np.std(x) for x in cf]
cf_cv = [np.std(x)/np.mean(x) for x in cf]

for i, x in enumerate(cf):
    plt.plot(positions[i], x)


#f = open("R:/APS 2019-1/KI4p8_cf.pckl", "wb")
#pickle.dump([positions[3], cf[3]], f)
#f.close()



