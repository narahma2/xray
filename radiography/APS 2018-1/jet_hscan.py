# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 14:01:43 2019

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
from skimage.transform import rotate
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from Statistics.CIs_LinearRegression import lin_fit

tests = glob.glob("R:/APS 2018-1/Imaging/Processed/Jets/*mm*.pckl")
tests.append(glob.glob("R:/APS 2018-1/Imaging/Processed/Jets/Jet_700-um_72-mS_.pckl")[0])

cf = len(tests) * [np.nan]
optical_diameters_avg = len(tests) * [np.nan]
model_epl_avg = len(tests) * [np.nan]
c1 = len(tests) * [np.nan]
c2 = len(tests) * [np.nan]
offset = len(tests)  * [np.nan]
positions = len(tests) * [np.nan]
hscan_position = len(tests) * [np.nan]
calibration_dataset = len(tests) * [np.nan]

#%%
for k, ss in enumerate(tests):        
    f = open(ss, "rb")
    processed_data = pickle.load(f)
    f.close()
    
    calibration_dataset[k] = len(processed_data) * [np.nan]
    
    for j, _ in enumerate(processed_data):
        calibration_dataset[k][j] = {"Axial Positions":  processed_data[j]["Axial Position"],
                                     "Lateral Positions": processed_data[j]["Lateral Position"], 
                                     "Optical Diameters": processed_data[j]["Optical Diameter"],
                                     "Model EPL Diameters": processed_data[j]["Model EPL Diameter"],
                                     "Offset": processed_data[j]["Offset EPL"]}
        
    #    break
    
    axial_positions = [x["Axial Positions"] for x in calibration_dataset[k]]
    axial_positions = [x for x in axial_positions for x in x]
    
    lateral_positions = [x["Lateral Positions"] for x in calibration_dataset[k]]
    lateral_positions = [x for x in lateral_positions for x in x]
    hscan_position[k] = np.nanmean(lateral_positions)
    
    optical_diameters = [x["Optical Diameters"] for x in calibration_dataset[k]]
    optical_diameters = [x for x in optical_diameters for x in x]
    
    model_epl = [x["Model EPL Diameters"] for x in calibration_dataset[k]]
    model_epl = [x for x in model_epl for x in x]
    
    offset[k] = np.mean([x["Offset"] for x in calibration_dataset[k]])
    
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

cf_mean = [np.nanmean(x) for x in cf]
cf_std = [np.nanstd(x) for x in cf]
cf_cv = [np.nanstd(x)/np.nanmean(x) for x in cf]

diameters = [x.rsplit('_')[1].rsplit('-')[0].rsplit('b')[0] for x in tests]
diameters = np.array(diameters, dtype=int)
KI_conc = [x.rsplit('_')[2].rsplit('-')[0] for x in tests]
KI_conc = ["0" if x=="0p0" else x for x in KI_conc]
KI_conc = ["11.5" if x=="11p5" else x for x in KI_conc]
KI_conc = np.array(KI_conc, dtype=float)

test_summary = pd.DataFrame({"Location": hscan_position, "Diameter": diameters, "KI %": KI_conc, "CF": [(round(x, 3), round(y, 3)) for x,y in zip(cf_mean, cf_std)], 
                             "Offset": offset})

#%% Plots
dark = np.array(Image.open("R:/APS 2018-1/Imaging/Processed/Jets/AVG_Jet_dark2.tif"))
#dark = rotate(dark, 2.0)

flatfieldavg = np.array(Image.open("R:/APS 2018-1/Imaging/Processed/Jets/AVG_Jet_flat2.tif"))
#flatfieldavg = rotate(flatfieldavg, 2.0)
flatfieldavg_darksub = flatfieldavg - dark

beam_middle = np.zeros((flatfieldavg_darksub.shape[0]))
for i in range(flatfieldavg_darksub.shape[0]):
    warnings.filterwarnings("ignore")
    beam_middle[i] = np.argmax(savgol_filter(flatfieldavg_darksub[i,:], 55, 3))
    warnings.filterwarnings("default")

beam_middle_avg = int(np.mean(beam_middle).round())
 
# Beam variation
plt.figure()
plt.plot(np.linspace(1, 768, 768) - beam_middle_avg, np.mean(flatfieldavg_darksub, axis=0) / max(np.mean(flatfieldavg_darksub, axis=0)), label='Beam Variation')
plt.title('Horizontal White Beam Variation')
plt.xlabel('Horizontal Location [px]')
plt.ylabel('Normalized Intensity [-]')
plt.ylim([0, 1])

# CF plot
plt.figure()
plt.scatter(np.array(test_summary['Location'])-beam_middle_avg, [x[0] for x in test_summary['CF']])
plt.title('Scatter Plot of CF vs. Horizontal Pixel Location')
plt.xlabel('Horizontal Location [px]')
plt.ylabel('CF [-]')
plt.ylim([0.4, 1.4])
        
        
        
        
        
        
        
        