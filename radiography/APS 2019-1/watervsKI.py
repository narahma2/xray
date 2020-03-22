# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:15:45 2019

@author: rahmann
"""

import os
import h5py
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths

#%% Calculate average
test_matrix = pd.read_csv("R:/APS 2019-1/test_matrix.txt", sep="\t+", engine="python")
spray_conditions = ["water", "KI"]
spray_indices = [24, 35]

f = open("R:/APS 2019-1/water_model.pckl", "rb")
water_model = pickle.load(f)
f.close()

save_folder = "R:/APS 2019-1/Water_vs_KI/"

for i, x in enumerate(spray_indices):
    if i == 0:
        f = open("R:/APS 2019-1/Imaging/Calibration2_x0_y3_corrected.pckl", "rb")
    else:
        f = open("R:/APS 2019-1/Imaging/KI_Liquid_Jet_S001_corrected.pckl", "rb")
    _, positions, cf, cs = pickle.load(f)
    f.close()
    
    test_name = test_matrix["Test"][spray_indices[i]]
    test_path = "R:\\APS 2019-1\\Imaging\\Raw_WB_Images\\" + test_name
    tests = glob.glob(test_path + "\\*.tif")
    
    cm_pix = 20 / 2.1 / (1000*10)
    dark = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_dark_current.tif'))
    flatfield = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_Background_NewYAG.tif'))
    flatfield_darksub = flatfield - dark
    
    beam_middle = np.zeros((flatfield_darksub.shape[1]))
    for j in range(flatfield_darksub.shape[1]):
        beam_middle[j] = np.argmax(savgol_filter(flatfield_darksub[:,j], 55, 3))
    
    beam_middle_avg = int(np.mean(beam_middle).round())
    angles = water_model[0]
    angle_to_px = [beam_middle_avg+3500*np.tan(x*10**-3)/cm_pix for x in angles]
    
    ss = np.zeros((512,768))
    spray_middle = np.zeros((512,1), dtype=float)
    for ii, tt in enumerate(tests):
        data = np.array(Image.open(tt))
        data_norm = (data-dark) / flatfield_darksub
        offset = 1 - np.median(data_norm[100:400, 30:100])
        
        data_epl = np.empty(np.shape(data_norm), dtype=float)
        
        for z, k in enumerate(positions):
            j = np.argmin(abs(k-np.array(angle_to_px)))
            data_epl[int(k), :] = cs[j](data_norm[int(k), :] + offset)* cf[z] * 10
        
        data_epl = np.fliplr(data_epl)
        offset_epl = np.median(data_epl[100:400, 30:100])
        data_epl -= offset_epl
        
        ss += data_epl
        
        for n, m in enumerate(data_epl):
            if n >= 35:
                spray_middle[n] = spray_middle[n] + np.argmax(savgol_filter(m[100:-10], 25, 3)) + 99
        
        # Save the 1000th slice to compare non-correlated scans
        if ii == 1000:
            data_epl_1000 = data_epl
        
    averaged_epl = ss / len(tests)
    spray_middle /= len(tests)
    
    f = open(save_folder + spray_conditions[i] + "_epl.pckl", "wb")
    pickle.dump([averaged_epl, data_epl_1000], f)
    f.close()
    
#%% Plots
f = open(save_folder + "water_epl.pckl", "rb")
[water_averaged, water_1000] = pickle.load(f)
f.close()

f = open(save_folder + "KI_epl.pckl", "rb")
[KI_averaged, KI_1000] = pickle.load(f)
f.close()

xx = np.linspace(1,768,768)
xx = xx * cm_pix * 10       # in mm
xx = xx - np.mean(xx)

yy = np.linspace(0,511,512)
yy = yy * (2.3/251)         # in mm

# Time averaged plots
plt.figure()
plt.plot(xx[0::5], 1000*water_averaged[188][0::5], color='b', marker='o', fillstyle='none', label='Water Averaged')
plt.plot(xx[0::5], 1000*KI_averaged[188][0::5], color='g', marker='^', fillstyle='none', label='KI Averaged')
plt.title('Time Averaged Comparison at y/d = 0.75')
plt.xlim([-3, 3])
plt.ylim([-200, 2500])
plt.xlabel('Horizontal Location [mm]')
plt.ylabel('Equivalent Path Length [$\mu$m]')
plt.legend()
plt.savefig(save_folder + "timeaveraged_yd075.png")

plt.figure()
plt.plot(xx[0::5], 1000*water_averaged[314][0::5], color='b', marker='o', fillstyle='none', label='Water Averaged')
plt.plot(xx[0::5], 1000*KI_averaged[314][0::5], color='g', marker='^', fillstyle='none', label='KI Averaged')
plt.title('Time Averaged Comparison at y/d = 1.25')
plt.xlim([-3, 3])
plt.ylim([-200, 2500])
plt.xlabel('Horizontal Location [mm]')
plt.ylabel('Equivalent Path Length [$\mu$m]')
plt.legend()
plt.savefig(save_folder + "timeaveraged_yd125.png")

plt.figure()
plt.plot(xx[0::5], 1000*water_averaged[439][0::5], color='b', marker='o', fillstyle='none', label='Water Averaged')
plt.plot(xx[0::5], 1000*KI_averaged[439][0::5], color='g', marker='^', fillstyle='none', label='KI Averaged')
plt.title('Time Averaged Comparison at y/d = 1.75')
plt.xlim([-3, 3])
plt.ylim([-200, 2500])
plt.xlabel('Horizontal Location [mm]')
plt.ylabel('Equivalent Path Length [$\mu$m]')
plt.legend()
plt.savefig(save_folder + "timeaveraged_yd175.png")

plt.figure()
plt.plot(yy[54:-30][0::5], 1000*water_averaged[54:-30, 384][0::5], color='b', marker='o', fillstyle='none', label='Water Averaged')
plt.plot(yy[54:-30][0::5], 1000*KI_averaged[54:-30, 384][0::5], color='g', marker='^', fillstyle='none', label='KI Averaged')
plt.title('Time Averaged Comparison Down Centerline')
plt.xlim([yy[50], yy[-25]])
plt.ylim([1000, 3500])
plt.xlabel('Vertical Location [mm]')
plt.ylabel('Equivalent Path Length [$\mu$m]')
plt.legend()
plt.savefig(save_folder + "timeaveraged_centerline.png")

# Time resolved plots at 1000th slice
plt.figure()
plt.plot(xx[0::5], 1000*water_1000[188][0::5], color='b', marker='o', fillstyle='none', label='Water Averaged')
plt.plot(xx[0::5], 1000*KI_1000[188][0::5], color='g', marker='^', fillstyle='none', label='KI Averaged')
plt.title('Non-Time-Correlated Comparison at y/d = 0.75')
plt.xlim([-3, 3])
plt.ylim([-300, 5000])
plt.xlabel('Horizontal Location [mm]')
plt.ylabel('Equivalent Path Length [$\mu$m]')
plt.legend()
plt.savefig(save_folder + "timeresolved_yd075.png")

plt.figure()
plt.plot(xx[0::5], 1000*water_1000[314][0::5], color='b', marker='o', fillstyle='none', label='Water Averaged')
plt.plot(xx[0::5], 1000*KI_1000[314][0::5], color='g', marker='^', fillstyle='none', label='KI Averaged')
plt.title('Non-Time-Correlated Comparison at y/d = 1.25')
plt.xlim([-3, 3])
plt.ylim([-300, 5000])
plt.xlabel('Horizontal Location [mm]')
plt.ylabel('Equivalent Path Length [$\mu$m]')
plt.legend()
plt.savefig(save_folder + "timeresolved_yd125.png")

plt.figure()
plt.plot(xx[0::5], 1000*water_1000[439][0::5], color='b', marker='o', fillstyle='none', label='Water Averaged')
plt.plot(xx[0::5], 1000*KI_1000[439][0::5], color='g', marker='^', fillstyle='none', label='KI Averaged')
plt.title('Non-Time-Correlated Comparison at y/d = 1.75')
plt.xlim([-3, 3])
plt.ylim([-300, 5000])
plt.xlabel('Horizontal Location [mm]')
plt.ylabel('Equivalent Path Length [$\mu$m]')
plt.legend()
plt.savefig(save_folder + "timeresolved_yd175.png")

plt.figure()
plt.plot(yy[54:-30][0::5], 1000*water_1000[54:-30, 384][0::5], color='b', marker='o', fillstyle='none', label='Water Averaged')
plt.plot(yy[54:-30][0::5], 1000*KI_1000[54:-30, 384][0::5], color='g', marker='^', fillstyle='none', label='KI Averaged')
plt.title('Non-Time-Correlated Comparison Down Centerline')
plt.xlim([yy[50], yy[-25]])
plt.ylim([-300, 5000])
plt.xlabel('Vertical Location [mm]')
plt.ylabel('Equivalent Path Length [$\mu$m]')
plt.legend()
plt.savefig(save_folder + "timeresolved_centerline.png")









    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    