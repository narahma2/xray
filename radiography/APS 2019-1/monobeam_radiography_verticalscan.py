# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:40:01 2019

@author: rahmann
"""

import sys
sys.path.append('E:/OneDrive - purdue.edu/Research/GitHub/coding/python')

import os
import h5py
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from Spectra.spectrum_modeling import density_KIinH2O

#%%
test_conditions = pd.read_csv("R:/APS 2019-1/test_conditions.txt", sep="\t+", engine="python")
tests_grouped = test_conditions.groupby(by=['Imaging'])
tests_keys = list(tests_grouped.groups.keys())[4:]

folder = "R:\\APS 2019-1\\HDF5"

# Calculate offset from a previous horizontal scan
f = h5py.File(folder + '\\Scan_5129.hdf5', 'r')
BIM = np.array(f["BIM"])
PIN = np.array(f["PINDiode"])
extinction_length = np.log(BIM / PIN)
# Attenuation coefficient (photoelectric absorption - cm^2/g)
# Convert to mm^2/g as lengths are in mm, and then multiply by density in g/mm^3
# Pure water @ 8 keV - <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
atten_coeff = (9.919*(10*10))*(0.001)
EPL = extinction_length / atten_coeff
offset_mb = np.median(EPL[0:10,:])

for z in tests_keys:
   indices = [i for i,s in enumerate(list(tests_grouped.get_group(z)['Y Positions (mm)'])) if 'to' in s]
   temp_list = list(tests_grouped.get_group(z)['Scan'])
   EPL = len(indices) * [None]
   wbp = len(indices) * [None]
   xp = len(indices) * [None]
   scan = len(indices) * [None]
   y = len(indices) * [None]
   for n, yy in enumerate(indices):
       scan[n] = temp_list[yy]
       f = h5py.File(folder + "\\Scan_" + str(scan[n]) + '.hdf5', 'r')
       xp[n] = np.array(f["X"]).mean()
       y[n] = np.array(f["Y"])
       BIM = np.array(f["BIM"])
       PIN = np.array(f["PINDiode"])
       extinction_length = np.log(BIM / PIN)
       
       EPL[n] = (extinction_length / atten_coeff) - offset_mb

   f = open("R:/APS 2019-1/Imaging/Calibration2_x0_y3_corrected.pckl", "rb")
   _, positions, cf, cs = pickle.load(f)
   f.close()

   f = open("R:/APS 2019-1/water_model.pckl", "rb")
   water_model = pickle.load(f)
   f.close()

   test_path = "R:/APS 2019-1/Imaging/Raw_WB_Images/" + z

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
   
   tests = glob.glob(test_path + "_S001\\*.tif")
   ss = np.zeros((512,768))
   spray_middle = np.zeros((512,1), dtype=float)
   for i, tt in enumerate(tests):
       data = np.array(Image.open(tt))
       data_norm = (data-dark) / flatfield_darksub
       offset = 1 - np.median(data_norm[100:400, 30:100])
        
       data_epl = np.empty(np.shape(data_norm), dtype=float)
        
       for zz, k in enumerate(positions):
           j = np.argmin(abs(k-np.array(angle_to_px)))
           data_epl[int(k), :] = cs[j](data_norm[int(k), :] + offset)* cf[zz] * 10
        
       data_epl = np.fliplr(data_epl)
       offset_epl = np.median(data_epl[100:400, 30:100])
       data_epl -= offset_epl
        
       ss += data_epl
       
       for n, m in enumerate(data_epl):
            if n >= 35:
                spray_middle[n] = spray_middle[n] + np.argmax(savgol_filter(m[100:-10], 25, 3)) + 99
       
       # Save the 1000th slice to compare non-correlated scans
       if i == 1000:
           data_epl_1000 = data_epl

   yy = np.linspace(0,511,512)
   yy = yy * (2.3/251)       # in mm
   
   break
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   