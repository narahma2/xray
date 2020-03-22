# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:00:32 2019

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
tests_keys = list(tests_grouped.groups.keys())[2:]

folder = "R:\\APS 2019-1\\HDF5"

for z in tests_keys:
   indices = [i for i,s in enumerate(list(tests_grouped.get_group(z)['Y Positions (mm)'])) if 'to' not in s]
   temp_list = list(tests_grouped.get_group(z)['Scan'])
   EPL = len(indices) * [None]
   wbp = len(indices) * [None]
   yp = len(indices) * [None]
   scan = len(indices) * [None]
   x = len(indices) * [None]
   for n, y in enumerate(indices):
       scan[n] = temp_list[y]
       f = h5py.File(folder + "\\Scan_" + str(scan[n]) + '.hdf5', 'r')
       x[n] = np.array(f["X"])
       BIM = np.array(f["BIM"])
       PIN = np.array(f["PINDiode"])
       extinction_length = np.log(BIM / PIN)
       
       wbp[n] = round(float(list(tests_grouped.get_group(z)['Y Positions (mm)'])[y]) / (2.3/251))
       yp[n] = float(list(tests_grouped.get_group(z)['Y Positions (mm)'])[y])
       
       # Attenuation coefficient (photoelectric absorption - cm^2/g)
       # Convert to mm^2/g as lengths are in mm, and then multiply by density in g/mm^3
       if "KI" not in z:
           # Pure water @ 8 keV - <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
           atten_coeff = (9.919*(10*10))*(0.001)
       elif "KI" in z:
           # 4.8% KI in water @ 8 keV <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
           atten_coeff = (21.68*(10*10))*(density_KIinH2O(4.8)/1000)
       
       EPL[n] = extinction_length / atten_coeff
       offset = np.median(EPL[n][0:10,:])
       EPL[n] = EPL[n] - offset
       
   if "KI" not in z:
       f = open("R:/APS 2019-1/Imaging/Calibration2_x0_y3_corrected.pckl", "rb")
   elif "KI" in z:
       f = open("R:/APS 2019-1/Imaging/KI_Liquid_Jet_S001_corrected.pckl", "rb")
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

   if "KI" not in z:
       tests = glob.glob(test_path + "_S001\\*.tif")
   elif "KI" in z:
       tests = glob.glob(test_path + "\\*.tif")
   ss = np.zeros((512,768))

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

   xx = np.linspace(1,768,768)
   xx = xx * cm_pix * 10       # in mm
   xx = xx - np.mean(xx)
    
   if not os.path.exists('R:/APS 2019-1/Monobeam/' + z):
       os.makedirs('R:/APS 2019-1/Monobeam/' + z)
    
   for n, _ in enumerate(indices):
       plt.figure()
       plt.plot(x[n], np.mean(EPL[n], axis=1), label='Monobeam Averaged')
       plt.plot(xx, ss[wbp[n], :] / i, label='Whitebeam Averaged')
       plt.xlim([-3, 3])
       plt.ylim([0, 5])
       plt.title(str(yp[n]) + ' mm Downstream')
       plt.legend()
       plt.savefig('R:/APS 2019-1/Monobeam/' + z + '/comparison_' + str(yp[n]) + 'mm.png')
    
   plt.figure()
   plt.imshow(data_epl, vmin=0, vmax=5)
   plt.colorbar()
   plt.title('EPL [mm] Mapping of Spray')
   for n, _ in enumerate(indices):
       plt.plot(np.linspace(1,768,768), np.linspace(wbp[n], wbp[n], 768))
   plt.savefig('R:/APS 2019-1/Monobeam/' + z + '/Spray Image.png')
    
   f = open("R:/APS 2019-1/Monobeam/" + z + "/monobeam_data.pckl", "wb")
   pickle.dump([EPL, scan, wbp, yp], f)
   f.close()
