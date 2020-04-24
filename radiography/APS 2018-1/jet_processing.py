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
from skimage.transform import rotate
from Statistics.CIs_LinearRegression import run_all, lin_fit, conf_calc, ylines_calc, plot_linreg_CIs
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse, plot_widths
from timeit import default_timer as timer

def main():
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

		ratio_ellipse_folder = project_folder + '/Processed/' + scintillator + '/RatioEllipse'
		if not os.path.exists(ratio_ellipse_folder):
			os.makedirs(ratio_ellipse_folder)

		ratio_peak_folder = project_folder + '/Processed/' + scintillator + '/RatioPeak'
		if not os.path.exists(ratio_peak_folder):
			os.makedirs(ratio_peak_folder)

		vertical_ellipse_folder = project_folder + '/Processed/' + scintillator + '/EllipseVertical'
		if not os.path.exists(vertical_ellipse_folder):
			os.makedirs(vertical_ellipse_folder)

		vertical_peak_folder = project_folder + '/Processed/' + scintillator + '/PeakVertical'
		if not os.path.exists(vertical_peak_folder):
			os.makedirs(vertical_peak_folder)

		vertical_optical_folder = project_folder + '/Processed/' + scintillator + '/OpticalVertical'
		if not os.path.exists(vertical_optical_folder):
			os.makedirs(vertical_optical_folder)

		ratio_ellipseT_folder = project_folder + '/Processed/' + scintillator + '/RatioEllipseT'
		if not os.path.exists(ratio_ellipseT_folder):
			os.makedirs(ratio_ellipseT_folder)

		ratio_peakT_folder = project_folder + '/Processed/' + scintillator + '/RatioPeakT'
		if not os.path.exists(ratio_peakT_folder):
			os.makedirs(ratio_peakT_folder)

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
		cm_px = cm_px = 0.16 / 162   # See 'APS White Beam.xlsx -> Pixel Size'

		dark = np.array(Image.open(project_folder + '/Images/Uniform_Jets/Mean/AVG_Jet_dark2.tif'))
		flatfield = np.array(Image.open(project_folder + '/Images/Uniform_Jets/Mean/AVG_Jet_flat2.tif'))

		test_matrix = pd.read_csv(project_folder + '/APS White Beam.txt', sep='\t+', engine='python')
		#sl = [1, 8, 15, 44]

		for index, test_name in enumerate(test_matrix['Test']):
		#for index in sl:
			test_path = project_folder + '/Images/Uniform_Jets/Mean/AVG_' + test_name + '.tif'    
			model = models[KI_conc.index(test_matrix['KI %'][index])]
			TtoEPL = model[0] 
			EPLtoT = model[1]
			
			graph_folder = project_folder + '/Processed/' + scintillator + '/Graphs/' + test_name
			if not os.path.exists(graph_folder):
				os.makedirs(graph_folder)

			width_folder = project_folder + '/Processed/' + scintillator + '/Widths/' + test_name
			if not os.path.exists(width_folder):
				os.makedirs(width_folder)

			data = np.array(Image.open(test_path))
			data_norm = np.zeros(np.shape(data), dtype=float)
			warnings.filterwarnings('ignore')
			data_norm = (data - dark) / (flatfield - dark)
			warnings.filterwarnings('default')
			
			data_epl = np.zeros(np.shape(data_norm), dtype=float)
			cropped_view = np.linspace(start=20, stop=325, num=325-20+1, dtype=int)
			for _,k in enumerate(cropped_view):
				data_epl[k, :] = TtoEPL[k](data_norm[k, :])

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
			ellipse_epl = len(cropped_view) * [np.nan]
			optical_diameter = len(cropped_view) * [np.nan]
			peak_epl = len(cropped_view) * [np.nan]
			axial_position = len(cropped_view) * [np.nan]
			lateral_position = len(cropped_view) * [np.nan]
			fitted_graph = len(cropped_view) * [np.nan]
			epl_graph = len(cropped_view) * [np.nan]
			optical_T = len(cropped_view) * [np.nan]
			peak_T = len(cropped_view) * [np.nan]
			ellipse_T = len(cropped_view) * [np.nan]

			for z, k in enumerate(cropped_view):
				smoothed = savgol_filter(data_epl[k, :], 105, 7)
				warnings.filterwarnings('ignore')
				peaks, _ = find_peaks(smoothed, width=20, prominence=0.1)
				warnings.filterwarnings('default')
				
				if len(peaks) == 1:
					# Ellipse fitting
					warnings.filterwarnings('ignore')
					[relative_width, relative_max, lpos, rpos] = peak_widths(smoothed, peaks, rel_height=0.90)
					warnings.filterwarnings('default')
					relative_width = relative_width[0]
					relative_max = relative_max[0]
					left_bound[z] = lpos[0]
					right_bound[z] = rpos[0]
					ydata = smoothed[int(round(lpos[0])):int(round(rpos[0]))]
					warnings.filterwarnings('ignore')
					ellipse_epl[z], fitted_graph[z], epl_graph[z] = ideal_ellipse(ydata, relative_width, relative_max, cm_px)
					warnings.filterwarnings('default')
					optical_diameter[z] = relative_width * cm_px
					axial_position[z] = k
					lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))
					peak_epl[z] = smoothed[peaks[0]]
					# Convert diameters to transmissions
					optical_T[z] = EPLtoT[k](optical_diameter[z])
					peak_T[z] = EPLtoT[k](peak_epl[z])
					ellipse_T[z] = EPLtoT[k](ellipse_epl[z])
					
					# Plot the fitted and EPL graphs
					if z % 15 == 0:
						# Ellipse
						plot_ellipse(epl_graph[z], fitted_graph[z], graph_folder + '/' + scintillator + '_{0:03.0f}'.format(z) + '.png')
						plt.close()

						plot_widths(data_epl[k, :], peaks, relative_max, lpos[0], rpos[0], width_folder + '/' + scintillator + '_{0:03.0f}'.format(z) + '.png')
						plt.close()
			
			if len(peaks) == 1:
				signal = np.nanmean(data_epl[20:325, int(round(np.nanmean(lateral_position)))-20:int(round(np.nanmean(lateral_position)))+20])
			else:
				signal = 0
				
			noise = np.nanstd(data_epl[50:300, 25:125])
			SNR = signal / noise
			
			try:
				idx = np.isfinite(ellipse_epl) & np.isfinite(optical_diameter)
				linfits, linfits_err, r2_values = lin_fit(np.array(ellipse_epl)[idx], np.array(optical_diameter)[idx])
				
			except:
				pass
			
			processed_data = {'Diameters': [optical_diameter, ellipse_epl, peak_epl], 'Transmissions': [optical_T, ellipse_T, peak_T], 'Axial Position': axial_position,
								 'Lateral Position': lateral_position, 'Bounds': [left_bound, right_bound], 'Linear Fits': linfits, 
								 'Linear Fits Error': linfits_err, 'R2': r2_values, 'Offset EPL': offset_epl, 'SNR': SNR}

			with open(summary_folder + '/' + scintillator + '_' + test_name + '.pckl', 'wb') as f:
				pickle.dump(processed_data, f)

			# Boundary plot
			plt.figure()
			plt.imshow(data_epl, vmin=0, vmax=1, zorder=1)
			plt.scatter(x=left_bound, y=axial_position, s=1, color='red', zorder=2)
			plt.scatter(x=right_bound, y=axial_position, s=1, color='red', zorder=2)
			plt.scatter(x=lateral_position, y=axial_position, s=1, color='white', zorder=2)
			plt.title(test_name)
			plt.savefig('{0}/{1}.png'.format(boundary_folder, test_name))
			plt.close()

			# Vertical ellipse variation
			plt.figure()
			plt.plot(axial_position, np.array(ellipse_epl)*10, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Ellipse Diameter (mm)')
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(vertical_ellipse_folder, scintillator, test_name))
			plt.close()

			# Vertical peak variation
			plt.figure()
			plt.plot(axial_position, np.array(peak_epl)*10, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Peak Diameter (mm)')
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(vertical_peak_folder, scintillator, test_name))
			plt.close()

			# Vertical optical variation
			plt.figure()
			plt.plot(axial_position, np.array(optical_diameter)*10, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Optical Diameter (mm)')
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(vertical_optical_folder, scintillator, test_name))
			plt.close()

			# Vertical EPL ratio (ellipse)
			plt.figure()
			plt.plot(axial_position, np.array(ellipse_epl) / np.array(optical_diameter), ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Ellipse/Optical EPL Ratio')
			plt.ylim([2, 4])
			plt.savefig('{0}/{1}_{2}.png'.format(ratio_ellipse_folder, scintillator, test_name))
			plt.close()

			# Vertical EPL ratio (peak)
			plt.figure()
			plt.plot(axial_position, np.array(peak_epl) / np.array(optical_diameter), ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Peak/Optical EPL Ratio')
			plt.ylim([2, 4])
			plt.savefig('{0}/{1}_{2}.png'.format(ratio_peak_folder, scintillator, test_name))
			plt.close()

			# Vertical transmission ratio (ellipse)
			plt.figure()
			plt.plot(axial_position, np.array(ellipse_T) / np.array(optical_T), ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Ellipse/Optical Transmission Ratio')
			plt.ylim([0.6, 1])
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(ratio_ellipseT_folder, scintillator, test_name))
			plt.close()

			# Vertical transmission ratio (peak)
			plt.figure()
			plt.plot(axial_position, np.array(peak_T) / np.array(optical_T), ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Peak/Optical Transmission Ratio')
			plt.ylim([0.6, 1])
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(ratio_peakT_folder, scintillator, test_name))
			plt.close()

# Run this script
if __name__ == '__main__':
	main()