# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:11:16 2019

@author: rahmann
"""

import h5py
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths

#%%
#f = open("R:/APS 2019-1/water_cf.pckl", "rb")
#water_cf = pickle.load(f)
#f.close()

f = open("R:/APS 2019-1/KI4p8_model.pckl", "rb")
water_model = pickle.load(f)
f.close()

test_path = "R:/APS 2019-1/Imaging/Raw_WB_Images/KI_Liquid_Jet_S001"

cm_pix = 20 / 2.1 / (1000*10)

dark = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_dark_current.tif'))
flatfield = np.array(Image.open('R:/APS 2019-1/Imaging/Processed_WB_Images/AVG_Background_NewYAG.tif'))
flatfield_darksub = flatfield - dark

beam_middle = np.zeros((flatfield_darksub.shape[1]))
for i in range(flatfield_darksub.shape[1]):
    beam_middle[i] = np.argmax(savgol_filter(flatfield_darksub[:,i], 55, 3))

beam_middle_avg = int(np.mean(beam_middle).round())

angles = water_model[0]
angle_to_px = [beam_middle_avg+3500*np.tan(x*10**-3)/cm_pix for x in angles]

tests = glob.glob(test_path + "\\*.tif")

for i, tt in enumerate(tests):
    i += 1
    data = np.array(Image.open(tt))
    data_norm = (data-dark) / flatfield_darksub
    
    data_epl = np.empty(np.shape(data_norm), dtype=float)
    cropped_view = np.linspace(30, stop=480, num=480-30+1, dtype=int)
    
    for z, k in enumerate(cropped_view):
        j = np.argmin(abs(k-np.array(angle_to_px)))
        data_epl[k, :] = water_model[1][j](data_norm[k, :])
        
    break