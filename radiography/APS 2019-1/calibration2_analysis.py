# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:20:25 2019

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
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse

#%%
tests = glob.glob("R:/APS 2019-1/Imaging/Processed_WB_Images/Water/**/*.pckl", recursive=True)
sl = np.linspace(0,len(tests)-1,len(tests),dtype=int).tolist()
tests = [tests[index] for index in sl]
calibration_dataset = len(tests) * [None]
cf = len(sl) * [None]
c1 = len(sl) * [None]
c2 = len(sl) * [None]
offset = len(sl)  * [None]
SNR = len(sl) * [None]
positions = len(sl) * [None]
lateral_position = []
optical_diameters_avg = len(sl) * [None]
model_epl_avg = len(sl) * [None]

#injector_face = [203, 152, 101, 16, 255, 203, 152, 152, 101, 48, 16, 16, 255, 16, 203, 152, 101, 48, 16, 16]

names = [x.rsplit('Calibration2\\')[-1].rsplit('.')[0] for x in tests]

#%%
for i, test in enumerate(tests):
    f = open(test, "rb")
    processed_data = pickle.load(f)
    f.close()
    
    calibration_dataset[i] = len(processed_data) * [None]
    
    for j, _ in enumerate(processed_data):
        calibration_dataset[i][j] = {"Image": processed_data[j]["Image"].rsplit("Raw_WB_Images\\")[1],
                                     "Left Bound": processed_data[j]["Left Bound"],
                                     "Right Bound": processed_data[j]["Right Bound"],
                                     "Axial Positions":  processed_data[j]["Axial Position"],
                                     "Lateral Positions": processed_data[j]["Lateral Position"], 
                                     "Optical Diameters": processed_data[j]["Optical Diameter"],
                                     "Model EPL Diameters": processed_data[j]["Model EPL Diameter"],
                                     "Offset": processed_data[j]["Offset EPL"],
                                     "SNR": processed_data[j]["SNR"]}
        
    axial_positions = [x["Axial Positions"] for x in calibration_dataset[i]]
    axial_positions = [x for x in axial_positions for x in x]
    
    lateral_positions = [x["Lateral Positions"] for x in calibration_dataset[i]]
    lateral_positions = [x for x in lateral_positions for x in x]
    lateral_position.append(np.nanmean(lateral_positions))
    
    optical_diameters = [x["Optical Diameters"] for x in calibration_dataset[i]]
    optical_diameters = [x for x in optical_diameters for x in x]
    
    model_epl = [x["Model EPL Diameters"] for x in calibration_dataset[i]]
    model_epl = [x for x in model_epl for x in x]
    
    offset[i] = np.mean([x["Offset"] for x in calibration_dataset[i]])
    SNR[i] = np.mean([x["SNR"] for x in calibration_dataset[i]])
    
    summary = pd.DataFrame({"Axial Position": axial_positions, "Lateral Position": lateral_positions, 
                            "Optical Diameter": optical_diameters, "Model EPL Diameter": model_epl})
    
    summary = summary.sort_values(["Axial Position", "Lateral Position"])
    summary = summary.reset_index(drop=True)
    summary_list = [v for k, v in summary.groupby("Axial Position")]
    
    positions[i] = [np.mean(x["Axial Position"]) for x in summary_list]
    optical_diameters_avg[i] = [np.mean(x["Optical Diameter"]) for x in summary_list]
    model_epl_avg[i] = [np.mean(x["Model EPL Diameter"]) for x in summary_list]
    cf[i] = [x1/x2 for x1,x2 in zip(optical_diameters_avg[i], model_epl_avg[i])]
    
    summary = None
    summary_list = None

cf_mean = [np.mean(x) for x in cf]
cf_std = [np.std(x) for x in cf]
cf_cv = [np.std(x)/np.mean(x) for x in cf]

lb_errors = [(np.array(x) - ((np.array(cf[i]) - cf_std[i])*np.array(model_epl_avg[i]))) for i, x in enumerate(optical_diameters_avg)]
ub_errors = [(((np.array(cf[i]) + cf_std[i])*np.array(model_epl_avg[i])) - np.array(x)) for i, x in enumerate(optical_diameters_avg)]
plus_minus_errors = np.mean([np.array(lb_errors), np.array(ub_errors)], axis=0).tolist()

summary = pd.DataFrame({"Test": names, "CF": cf_mean, "Lateral Position": lateral_position, #"Injector Face": injector_face,
                        "Offset": offset, "SNR": SNR})

#%%
sl_top = [0,1,2,3,4,5]
for x in sl_top:
    plt.plot(positions[x], cf[x], label='x=' + str(round(summary['Lateral Position'][x])))
plt.legend()

master_position = np.linspace(0,511,512, dtype=int).tolist()
master_cf_mean = len(master_position) * [np.nan]
master_cf_std = len(master_position) * [np.nan]

for j, y in enumerate(master_position):
    xx = []
    for k, z in enumerate(sl_top):
        if y in positions[z]:
            xx.append(cf[z][positions[z].index(y)])
    master_cf_mean[j] = np.array(xx).mean()
    master_cf_std[j] = np.array(xx).std()

f = open("R:/APS 2019-1/water_cf.pckl", "wb")
pickle.dump([master_position, master_cf_mean], f)
f.close()

#%% EPL comparison
i = 3
j = 900
dark = 'R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_dark_current.tif'
flat = 'R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_Background_NewYAG.tif'
model_pckl = "R:/APS 2019-1/water_model.pckl"


data_epl = convert2EPL(test_path='R:\\APS 2019-1\\Imaging\\Raw_WB_Images\\' + calibration_dataset[i][j]["Image"], offset=calibration_dataset[i][j]["Offset"], model_pckl=model_pckl, cm_pix=20 / 2.1 / (1000*10), 
                       dark_path=dark, flat_path=flat, cropped_view=np.linspace(30,480,480-30+1))
ellipse(data_epl=master_cf_mean[200]*data_epl, position=200, cm_pix=20 / 2.1 / (1000*10), relative_height=0.8, plot=True) 










































