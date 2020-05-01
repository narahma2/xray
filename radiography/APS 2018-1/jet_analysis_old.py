# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:45:47 2019

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
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse
    
#%%

test_type = ["2000um, 0% KI", "2000um, 1.6% KI", "2000um, 3.4% KI", "2000um, 4.8% KI", "2000um, 8.0% KI", "2000um, 10.0% KI", "2000um, 11.1% KI",
             "700um, 0%  KI",  "700um, 1.6%", 
             "700um, 3.4% KI", "700um, 4.8% KI", "700um, 8.0% KI", "700um, 10.0% KI", "700um, 11.1% KI"]
tests = glob.glob("R:/APS 2018-1/Imaging/Processed/Jets/*.pckl")
#sl = [[0,7], [1,8], [2,9], [3,10], [4, 11], [5, 12], [6, 13]]
sl = [0,7, 1,8, 2,9, 3,10, 4,11, 5,12, 6,13, 14, 15,44, 
      16,45, 17,46, 18,47, 19,48, 43,49]
test_type_ind = [[0,7], [1,8], [2,9], [3,10], [4,11], [5,12], [6,13], [14], [15,44], 
                 [16,45], [17,46], [18,47], [19,48], [43,49]]
#sl = sl[0:2]

cf = len(sl) * [None]
optical_diameters_avg = len(sl) * [None]
model_epl_avg = len(sl) * [None]
c1 = len(sl) * [None]
c2 = len(sl) * [None]
offset = len(sl)  * [None]
SNR = len(sl) * [None]
positions = len(sl) * [None]
calibration_dataset = len(sl) * [None]

#%%
for k, ss in enumerate(sl):    
    tt = tests[ss]
    
    f = open(tt, "rb")
    processed_data = pickle.load(f)
    f.close()
    
    calibration_dataset[k] = len(processed_data) * [None]
    
    for j, _ in enumerate(processed_data):
            calibration_dataset[k][j] = {"Image": processed_data[j]["Image"],
                                         "Axial Positions":  processed_data[j]["Axial Position"],
                                         "Lateral Positions": processed_data[j]["Lateral Position"], 
                                         "Optical Diameters": processed_data[j]["Optical Diameter"],
                                         "Model EPL Diameters": processed_data[j]["Model EPL Diameter"],
                                         "C1": processed_data[j]["Linear Fits"].c[0],
                                         "C2": processed_data[j]["Linear Fits"].c[1],
                                         "Offset": processed_data[j]["Offset EPL"],
                                         "SNR": processed_data[j]["SNR"]}
        
    #    break
    
    axial_positions = [x["Axial Positions"] for x in calibration_dataset[k]]
    axial_positions = [x for x in axial_positions for x in x]
    
    lateral_positions = [x["Lateral Positions"] for x in calibration_dataset[k]]
    lateral_positions = [x for x in lateral_positions for x in x]
    
    optical_diameters = [x["Optical Diameters"] for x in calibration_dataset[k]]
    optical_diameters = [x for x in optical_diameters for x in x]
    
    model_epl = [x["Model EPL Diameters"] for x in calibration_dataset[k]]
    model_epl = [x for x in model_epl for x in x]
    
    c1[k] = [x["C1"] for x in calibration_dataset[k]]
    c2[k] = [x["C2"] for x in calibration_dataset[k]]
    
    offset[k] = np.mean([x["Offset"] for x in calibration_dataset[k]])
    SNR[k] = np.mean([x["SNR"] for x in calibration_dataset[k]])
    
    summary = pd.DataFrame({"Axial Position": axial_positions, "Lateral Position": lateral_positions, 
                            "Optical Diameter": optical_diameters, "Model EPL Diameter": model_epl})
    
    summary = summary.sort_values(["Axial Position", "Lateral Position"])
    summary = summary.reset_index(drop=True)
    summary_list = [v for k, v in summary.groupby("Axial Position")]
    
    positions[k] = [np.mean(x["Axial Position"]) for x in summary_list][5:]
    optical_diameters_avg[k] = [np.mean(x["Optical Diameter"]) for x in summary_list]
    model_epl_avg[k] = [np.mean(x["Model EPL Diameter"]) for x in summary_list]
    cf[k] = [x1/x2 for x1,x2 in zip(optical_diameters_avg[k], model_epl_avg[k])][5:]
    
    summary = None
    summary_list = None
    
#cf.remove([])
#cf.remove([])
    
c1_mean = [np.mean(x) for x in c1]
c1_std = [np.std(x) for x in c1]
c1_cv = [np.std(x)/np.mean(x) for x in c1]

c2_mean = [np.mean(x) for x in c2]
c2_std = [np.std(x) for x in c2]
c2_cv = [np.std(x)/np.mean(x) for x in c2]

cf_mean = [np.mean(x) for x in cf]
cf_std = [np.std(x) for x in cf]
cf_cv = [np.std(x)/np.mean(x) for x in cf]

diameters = [x.rsplit('_')[1].rsplit('-')[0].rsplit('b')[0] for x in tests]
diameters = np.array(diameters, dtype=int)
salinity = [x.rsplit('_')[2].rsplit('-')[0] for x in tests]
salinity = ["0" if x=="0p0" else x for x in salinity]
salinity = ["11.5" if x=="11p5" else x for x in salinity]
salinity = np.array(salinity, dtype=float)

lb_errors = [np.mean(np.array(x[5:]) - ((np.array(cf[i]) - cf_std[i])*np.array(model_epl_avg[i][5:]))) for i, x in enumerate(optical_diameters_avg)]
ub_errors = [np.mean(((np.array(cf[i]) + cf_std[i])*np.array(model_epl_avg[i][5:])) - np.array(x[5:])) for i, x in enumerate(optical_diameters_avg)]
plus_minus_errors = np.mean([np.array(lb_errors), np.array(ub_errors)], axis=0).tolist()

test_summary = pd.DataFrame({"Slice": sl, "Diameter": diameters[sl], "Salinity": salinity[sl], "CF": [(round(x, 3), round(y, 3)) for x,y in zip(cf_mean, cf_std)], 
                             "C1": [(round(x, 3), round(y, 3)) for x,y in zip(c1_mean, c1_std)], "C2": [(round(x, 3), round(y, 3)) for x,y in zip(c2_mean, c2_std)],
                              "Offset": offset, "SNR": SNR})

#%% Error summary
master_position = np.linspace(1,340,340, dtype=int).tolist()
master_cf_mean = 7 * [None]
master_cf_std = 7 * [None]
master_lb_errors = 7 * [None]
master_ub_errors = 7 * [None]

KI_conc = [0.0, 1.6, 3.4, 4.8, 8.0, 10.0, 11.1]
KI_ind = [[0,1,14], [2,3,15,16], [4,5,17,18], [6,7,19,20], [8,9,21,22], [10,11,23,24], [12,13,25,26]]

for i, x in enumerate(master_cf_mean):
    master_cf_mean[i] = len(master_position) * [np.nan]
    master_cf_std[i] = len(master_position) * [np.nan]
    for j, y in enumerate(master_position):
        xx = []
        for k, z in enumerate(KI_ind[i]):
            if y in positions[z]:
                xx.append(cf[z][positions[z].index(y)])
        master_cf_mean[i][j] = np.array(xx).mean()
        master_cf_std[i][j] = np.array(xx).std()
     
for i, x in enumerate(master_lb_errors):
    master_lb_errors[i] = len(master_position) * [np.nan]
    master_ub_errors[i] = len(master_position) * [np.nan]
    for j, y in enumerate(master_position):
        master_lb_errors[i][j] = len(KI_ind[i]) * [np.nan]
        master_ub_errors[i][j] = len(KI_ind[i]) * [np.nan]
        for k, z in enumerate(KI_ind[i]):
            if y in positions[z]:
                master_lb_errors[i][j][k] = abs(optical_diameters_avg[z][positions[z].index(y)+5] - (master_cf_mean[i][j] - 
                                                master_cf_std[i][j])*model_epl_avg[z][positions[z].index(y)+5])
                master_ub_errors[i][j][k] = abs((master_cf_mean[i][j] - master_cf_std[i][j])*model_epl_avg[z][positions[z].index(y)+5] - 
                                                optical_diameters_avg[z][positions[z].index(y)+5])
    
for i, x in enumerate(master_lb_errors):
    plt.figure()
    for j, y in enumerate(KI_ind[i]):
        plt.plot([x[j]*10*1000 for x in master_lb_errors[i]], label=str(test_summary['Diameter'][y]) + 'um (' + str(y) + ')')
        plt.legend()
        plt.title(str(KI_conc[i]) + '% KI Jet Errors')
        plt.ylabel('Error [um]')
        plt.xlabel('Vertical Location [px]')
    plt.savefig('R:\\APS 2018-1\\Imaging\\Processed\\Error Plots\\KI_' + str(KI_conc[i]) + '%.png')
    
plt.figure()
for i, x in enumerate(master_lb_errors):
    plt.subplot(2,4,i+1)
    for j, y in enumerate(KI_ind[i]):
        plt.plot([x[j]*10*1000 for x in master_lb_errors[i]], label=str(test_summary['Diameter'][y]) + 'um (' + str(y) + ')')
        plt.legend()
        plt.title(str(KI_conc[i]) + '% KI Jet Errors')
        plt.ylabel('Error [um]')
        plt.xlabel('Vertical Location [px]')
plt.savefig('R:\\APS 2018-1\\Imaging\\Processed\\Error Plots\\allcurves.png', dpi=100)

#%% EPL comparison
i = 0
j = 900
dark = "R:/APS 2018-1/Imaging/Processed/Jets/AVG_Jet_dark2.tif"
flatfields = glob.glob("R:\\APS 2018-1\\Imaging\\Jets Big and Day\\Jet_flat2\\*.tif")
flat = flatfields[j]
model_KI = [0, 11.5, 24, 34, 57, 72, 80]
models = ['water', 'KI1p6', 'KI3p4', 'KI4p8', 'KI8p0', 'KI10p0', 'KI11p1']
model_pckl = "R:/APS 2018-1/Imaging/" + models[model_KI.index(salinity[i])] + "_model.pckl"

data_epl = convert2EPL(test_path=calibration_dataset[i][j]["Image"], offset=calibration_dataset[i][j]["Offset"], model_pckl=model_pckl, cm_pix=700/84 / (1000*10), 
                       dark_path=dark, flat_path=flat, cropped_view=np.linspace(1,340,340))
ellipse(data_epl=data_epl, position=200, cm_pix=700/84 / (1000*10), relative_height=0.8, plot=True) 
data_epl_corrected = np.array([x * master_cf_mean[i][k] for k,x in enumerate(data_epl[0:340,:])])
ellipse(data_epl=data_epl_corrected, position=200, cm_pix=700/84 / (1000*10), relative_height=0.8, plot=True)            

#%% KI Curves
KI_conc = [0.0, 1.6, 3.4, 4.8, 8.0, 10.0, 11.1]

for j, y in enumerate([0, 11.5, 24, 34, 57, 72, 80]):
    plt.figure()
    for x in test_summary[test_summary["Salinity"].isin([y])].index:
        plt.plot(positions[x], cf[x], label=str(test_summary['Diameter'][x]) + 'um (' + str(x) + '), ' + str(test_summary["Salinity"][x]) + '% KI')
    plt.plot(master_position, savgol_filter(master_cf_mean[j], 55, 3), color='k', label='Average')
    plt.plot(master_position, savgol_filter(master_cf_mean[j], 55, 3) - savgol_filter(master_cf_std[j], 55, 3), color='k', linestyle='-.')
    plt.plot(master_position, savgol_filter(master_cf_mean[j], 55, 3) + savgol_filter(master_cf_std[j], 55, 3), color='k', linestyle='-.')
    plt.legend()
    plt.title(str(KI_conc[j]) + '% KI Curves')
    plt.xlabel('Vertical Location [px]')
    plt.ylabel('CF [-]')
    plt.ylim([0.4, 1.4])
    plt.savefig('R:\\APS 2018-1\\Imaging\\Processed\\KI Curves\\KI_' + str(KI_conc[j]) + '%.png')
    
for j, y in enumerate([0, 11.5, 24, 34, 57, 72, 80]):
    plt.subplot(2,4,j+1)
    for x in test_summary[test_summary["Salinity"].isin([y])].index:
        plt.plot(positions[x], cf[x], label=str(test_summary['Diameter'][x]) + 'um (' + str(x) + '), ' + str(test_summary["Salinity"][x]) + '% KI')
    plt.plot(master_position, savgol_filter(master_cf_mean[j], 55, 3), color='k', label='Average')
    plt.plot(master_position, savgol_filter(master_cf_mean[j], 55, 3) - savgol_filter(master_cf_std[j], 55, 3), color='k', linestyle='-.')
    plt.plot(master_position, savgol_filter(master_cf_mean[j], 55, 3) + savgol_filter(master_cf_std[j], 55, 3), color='k', linestyle='-.')
    plt.legend()
    plt.title(str(KI_conc[j]) + '% KI Curves')
    plt.xlabel('Vertical Location [px]')
    plt.ylabel('CF [-]')
    plt.ylim([0.4, 1.4])
plt.savefig('R:\\APS 2018-1\\Imaging\\Processed\\KI Curves\\allcurves.png') 

#%% Diameter comparison scatter plots
for y in [0, 11.5, 24, 34, 57, 72, 80]:
    plt.figure()
    for x in test_summary[test_summary["Salinity"].isin([y])].index:
        plt.scatter(model_epl_avg[x], optical_diameters_avg[x], label=str(test_summary['Diameter'][x]) + 'um (' + str(x) + '), ' + str(test_summary["Salinity"][x]) + '% KI')
    plt.legend()
    plt.title(str(test_summary["Salinity"][x]) + '% Diameter Comparison')
    plt.xlabel('Avg. Model EPL [cm]')
    plt.ylabel('Avg. Optical Diameter [cm]')
#    plt.ylim([0.4, 1.4])
#    plt.savefig('R:\\APS 2018-1\\Imaging\\Processed\\KI Curves\\KI_' + str(y) + '%.png')      
            
            
            
            

