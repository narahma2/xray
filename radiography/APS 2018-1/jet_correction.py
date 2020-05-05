"""
-*- coding: utf-8 -*-
Re-processes the jets with the correction factors.

@Author: rahmann
@Date:   2020-05-01 19:47:14
@Last Modified by:   rahmann
@Last Modified time: 2020-05-01 19:47:14
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

import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from skimage.transform import rotate
from Statistics.CIs_LinearRegression import run_all, lin_fit, conf_calc, ylines_calc, plot_linreg_CIs
from Statistics.calc_statistics import rmse, mape, zeta, mdlq
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse, plot_widths
from timeit import default_timer as timer
from jet_processing import create_folder, process_jet

def main():
	# Location of APS 2018-1 data
	project_folder = sys_folder + '/X-ray Radiography/APS 2018-1/'

	#%% Imaging setup
	cm_px = 0.16 / 162   # See 'APS White Beam.xlsx -> Pixel Size'

	test_matrix = pd.read_csv(project_folder + '/APS White Beam.txt', sep='\t+', engine='python')

	# Scintillator
	scintillators = ['LuAG', 'YAG']

	# Correction factors
	correction_factors = ['peakT', 'ellipseT']

	for correction_factor in correction_factors:
		for scintillator in scintillators:
			#%% Load models from the whitebeam_2018-1 script
			f = open(project_folder + '/Model/water_model_' + scintillator + '.pckl', 'rb')
			water_model = pickle.load(f)
			f.close()
			
			f = open(project_folder + '/Model/KI1p6_model_' + scintillator + '.pckl', 'rb')
			KI1p6_model = pickle.load(f)
			f.close()

			f = open(project_folder + '/Model/KI3p4_model_' + scintillator + '.pckl', 'rb')
			KI3p4_model = pickle.load(f)
			f.close()

			f = open(project_folder + '/Model/KI4p8_model_' + scintillator + '.pckl', 'rb')
			KI4p8_model = pickle.load(f)
			f.close()

			f = open(project_folder + '/Model/KI8p0_model_' + scintillator + '.pckl', 'rb')
			KI8p0_model = pickle.load(f)
			f.close()

			f = open(project_folder + '/Model/KI10p0_model_' + scintillator + '.pckl', 'rb')
			KI10p0_model = pickle.load(f)
			f.close()

			f = open(project_folder + '/Model/KI11p1_model_' + scintillator + '.pckl', 'rb')
			KI11p1_model = pickle.load(f)
			f.close()

			# Top-level save folder
			save_folder = '{0}/Corrected/{1}_{2}/'.format(project_folder, scintillator, correction_factor)

			# Load in corresponding correction factor
			cf = np.loadtxt('{0}/Processed/{1}/{1}_{2}_cf.txt'.format(project_folder, scintillator, correction_factor))
			
			KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]
			models = [water_model, KI1p6_model, KI3p4_model, KI4p8_model, KI8p0_model, KI10p0_model, KI11p1_model]

			for index, test_name in enumerate(test_matrix['Test']):
			#for index in sl:
				test_path = '{0}/Processed/Normalized/Norm_{1}.tif'.format(project_folder, test_name)   
				model = models[KI_conc.index(test_matrix['KI %'][index])]
				nozzleD = test_matrix['Nozzle Diameter (um)'][index]
				KIperc = test_matrix['KI %'][index]
				TtoEPL = model[0] 
				EPLtoT = model[1]
				
				# Offset bounds found in ImageJ, X and Y are flipped!
				offset_sl_x = slice(test_matrix['BY'][index], test_matrix['BY'][index] + test_matrix['Height'][index])
				offset_sl_y = slice(test_matrix['BX'][index], test_matrix['BX'][index] + test_matrix['Width'][index])

				# Load in normalized images
				data_norm = np.array(Image.open(test_path))

				# Apply corresponding correction factor (based on KI %) to the normalized image
				data_norm /= cf[KI_conc.index(test_matrix['KI %'][index])]

				# Process the jet file
				process_jet(cm_px, save_folder, scintillator, index, test_name, test_path, TtoEPL, EPLtoT, offset_sl_x, offset_sl_y, data_norm)


# Run this script
if __name__ == '__main__':
	main()
