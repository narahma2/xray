# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:04:28 2019

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

import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy import optimize
from skimage.transform import rotate
from Statistics.CIs_LinearRegression import run_all, lin_fit, conf_calc, ylines_calc, plot_linreg_CIs
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse
from timeit import default_timer as timer

# Location of APS 2018-1 data
project_folder = sys_folder + '/X-ray Radiography/APS 2018-1/'

# Scintillator
scintillators = ['LuAG', 'YAG']

for scintillator in scintillators:
	epl_folder = project_folder + '/Processed/' + scintillator + '/EPL'
	if not os.path.exists(epl_folder):
		os.makedirs(epl_folder)

	summary_folder = project_folder + '/Processed/' + scintillator + '/Summary'
	if not os.path.exists(summary_folder):
		os.makedirs(summary_folder)

	diameters_folder = project_folder + '/Processed/' + scintillator + '/DiameterComparison'
	if not os.path.exists(diameters_folder):
		os.makedirs(diameters_folder)

	vertical_model_folder = project_folder + '/Processed/' + scintillator + '/ModelVertical'
	if not os.path.exists(vertical_model_folder):
		os.makedirs(vertical_model_folder)

	vertical_optical_folder = project_folder + '/Processed/' + scintillator + '/OpticalVertical'
	if not os.path.exists(vertical_optical_folder):
		os.makedirs(vertical_optical_folder)

	boundary_folder = project_folder + '/Processed/' + scintillator + '/Boundaries'
	if not os.path.exists(boundary_folder):
		os.makedirs(boundary_folder)

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

	KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]
	models = [water_model, KI1p6_model, KI3p4_model, KI4p8_model, KI8p0_model, KI10p0_model, KI11p1_model]

	#%% Imaging setup
	cm_pix = cm_pix = 0.16 / 162   # See 'APS White Beam.xlsx -> Pixel Size'

	dark = np.array(Image.open(project_folder + '/Images/Uniform_Jets/Mean/AVG_Jet_dark2.tif'))
	flatfield = np.array(Image.open(project_folder + '/Images/Uniform_Jets/Mean/AVG_Jet_flat2.tif'))

	test_matrix = pd.read_csv(project_folder + '/APS White Beam.txt', sep='\t+', engine='python')
	#sl = [1, 8, 15, 44]

	for index, test_name in enumerate(test_matrix['Test']):
	#for index in sl:
		test_path = project_folder + '/Images/Uniform_Jets/Mean/AVG_' + test_name + '.tif'    
		model = models[KI_conc.index(test_matrix['KI %'][index])]
		
		graph_folder = project_folder + '/Processed/' + scintillator + '/Graphs/' + test_name
		if not os.path.exists(graph_folder):
			os.makedirs(graph_folder)

		data = np.array(Image.open(test_path))
		data_norm = np.zeros(np.shape(data), dtype=float)
		warnings.filterwarnings('ignore')
		data_norm = (data - dark) / (flatfield - dark)
		warnings.filterwarnings('default')
		
		data_epl = np.zeros(np.shape(data_norm), dtype=float)
		cropped_view = np.linspace(start=20, stop=325, num=325-20+1, dtype=int)
		for _,k in enumerate(cropped_view):
			data_epl[k, :] = model[0][k](data_norm[k, :])

		# Offset bounds found in ImageJ, X and Y are flipped!
		offset_sl_x = slice(test_matrix['BY'][index], test_matrix['BY'][index] + test_matrix['Height'][index])
		offset_sl_y = slice(test_matrix['BX'][index], test_matrix['BX'][index] + test_matrix['Width'][index])
		offset_epl = np.nanmedian(data_epl[offset_sl_x, offset_sl_y]) 		
			
		# Correct the EPL values
		data_epl -= offset_epl

		# Save EPL images
		im = Image.fromarray(data_epl)
		im.save(epl_folder + '/' + test_path.rsplit('/')[-1].replace('AVG', scintillator))

		left_bound = len(cropped_view) * [np.nan]
		right_bound = len(cropped_view) * [np.nan]
		model_epl = len(cropped_view) * [np.nan]
		optical_diameter = len(cropped_view) * [np.nan]
		axial_position = len(cropped_view) * [np.nan]
		lateral_position = len(cropped_view) * [np.nan]
		fitted_graph = len(cropped_view) * [np.nan]
		epl_graph = len(cropped_view) * [np.nan]

		for z, k in enumerate(cropped_view):
			smoothed = savgol_filter(data_epl[k, :], 105, 7)
			warnings.filterwarnings('ignore')
			peaks, _ = find_peaks(smoothed, width=20, prominence=0.1)
			warnings.filterwarnings('default')
			
			if len(peaks) == 1:
				warnings.filterwarnings('ignore')
				[relative_width, relative_max, lpos, rpos] = peak_widths(smoothed, peaks, rel_height=0.90)
				warnings.filterwarnings('default')
				relative_width = relative_width[0]
				relative_max = relative_max[0]
				left_bound[z] = lpos[0]
				right_bound[z] = rpos[0]
				warnings.filterwarnings('ignore')
				model_epl[z], fitted_graph[z], epl_graph[z] = ideal_ellipse(smoothed[int(round(lpos[0])):int(round(rpos[0]))], relative_width, relative_max, cm_pix)
				warnings.filterwarnings('default')
				optical_diameter[z] = relative_width * cm_pix
				axial_position[z] = k
				lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))

				# Plot the fitted and EPL graphs
				if z % 15 == 0:
					plot_ellipse(epl_graph[z], fitted_graph[z], graph_folder + '/' + scintillator + '_{0:03.0f}'.format(z) + '.png')
					plt.close()
		
		if len(peaks) == 1:
			signal = np.nanmean(data_epl[20:325, int(round(np.nanmean(lateral_position)))-20:int(round(np.nanmean(lateral_position)))+20])
		else:
			signal = 0
			
		noise = np.nanstd(data_epl[50:300, 25:125])
		SNR = signal / noise
		
		try:
			idx = np.isfinite(model_epl) & np.isfinite(optical_diameter)
			linfits, linfits_err, r2_values = lin_fit(np.array(model_epl)[idx], np.array(optical_diameter)[idx])
			
		except:
			pass
		
		processed_data = {'Optical Diameter': optical_diameter, 'Model EPL Diameter': model_epl, 'Axial Position': axial_position,
							 'Lateral Position': lateral_position, 'Left Bound': left_bound, 'Right Bound': right_bound, 'Linear Fits': linfits, 
							 'Linear Fits Error': linfits_err, 'R2': r2_values, 'Offset EPL': offset_epl, 'SNR': SNR}

		with open(summary_folder + '/' + scintillator + '_' + test_name + '.pckl', 'wb') as f:
			pickle.dump(processed_data, f)

		# Create plots
		plt.figure()
		plt.imshow(data_epl, vmin=0, vmax=1, zorder=1)
		plt.scatter(x=left_bound, y=axial_position, s=1, color='red', zorder=2)
		plt.scatter(x=right_bound, y=axial_position, s=1, color='red', zorder=2)
		plt.scatter(x=lateral_position, y=axial_position, s=1, color='white', zorder=2)
		plt.title(test_name)
		plt.savefig('{0}/{1}.png'.format(boundary_folder, test_name))
		plt.close()

		plt.figure()
		plt.plot(np.array(optical_diameter)*10, np.array(model_epl)*10, ' o')
		plt.xlabel('Optical Diameter (mm)')
		plt.ylabel('Model EPL (mm)')
		plt.title('R^2 = ' + str(round(r2_values, 2)))
		plt.savefig('{0}/{1}_{2}.png'.format(diameters_folder, scintillator, test_name))
		plt.close()

		plt.figure()
		plt.plot(axial_position, np.array(model_epl)*10, ' o')
		plt.xlabel('Axial Position (px)')
		plt.ylabel('Model EPL (mm)')
		plt.title(test_name)
		plt.savefig('{0}/{1}_{2}.png'.format(vertical_model_folder, scintillator, test_name))
		plt.close()

		plt.figure()
		plt.plot(axial_position, np.array(optical_diameter)*10, ' o')
		plt.xlabel('Axial Position (px)')
		plt.ylabel('Optical Diameter (mm)')
		plt.title(test_name)
		plt.savefig('{0}/{1}_{2}.png'.format(vertical_optical_folder, scintillator, test_name))
		plt.close()