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
from Statistics.calc_statistics import rmse, mape, zeta, mdlq
from White_Beam.wb_functions import convert2EPL, ellipse, ideal_ellipse, plot_ellipse, plot_widths
from timeit import default_timer as timer

def create_folder(new_folder):
	if not os.path.exists(new_folder):
		os.makedirs(new_folder)

	return new_folder

def main():
	# Location of APS 2018-1 data
	project_folder = sys_folder + '/X-ray Radiography/APS 2018-1/'

	# Scintillator
	scintillators = ['LuAG', 'YAG']

	for scintillator in scintillators:
		epl_folder = create_folder(project_folder + '/Processed/' + scintillator + '/EPL')
		summary_folder = create_folder(project_folder + '/Processed/' + scintillator + '/Summary')
		ratio_ellipse_folder = create_folder(project_folder + '/Processed/' + scintillator + '/RatioEllipse')
		ratio_peak_folder = create_folder(project_folder + '/Processed/' + scintillator + '/RatioPeak')
		vertical_ellipse_folder = create_folder(project_folder + '/Processed/' + scintillator + '/EllipseVertical')
		vertical_peak_folder = create_folder(project_folder + '/Processed/' + scintillator + '/PeakVertical')
		vertical_optical_folder = create_folder(project_folder + '/Processed/' + scintillator + '/OpticalVertical')
		ratio_ellipseT_folder = create_folder(project_folder + '/Processed/' + scintillator + '/RatioEllipseT')
		ratio_peakT_folder = create_folder(project_folder + '/Processed/' + scintillator + '/RatioPeakT')
		boundary_folder = create_folder(project_folder + '/Processed/' + scintillator + '/Boundaries')

		with open(project_folder + '/Processed/' + scintillator + '/summary.txt', 'w') as filetowrite:
			filetowrite.write('Diameter\tKI %\tHoriz. Pos.\tMean Ratio Peak\tCV Ratio Peak\tMean Ratio PeakT\tCV Ratio PeakT\tMean Ratio Ellipse\tCV Ratio Ellipse\tMean Ratio EllipseT\tCV Ratio EllipseT\n')

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
		cm_px = 0.16 / 162   # See 'APS White Beam.xlsx -> Pixel Size'

		dark = np.array(Image.open(project_folder + '/Images/Uniform_Jets/Mean/AVG_Jet_dark2.tif'))
		flatfield = np.array(Image.open(project_folder + '/Images/Uniform_Jets/Mean/AVG_Jet_flat2.tif'))

		test_matrix = pd.read_csv(project_folder + '/APS White Beam.txt', sep='\t+', engine='python')
		#sl = [1, 8, 15, 44]

		for index, test_name in enumerate(test_matrix['Test']):
		#for index in sl:
			test_path = project_folder + '/Images/Uniform_Jets/Mean/AVG_' + test_name + '.tif'    
			model = models[KI_conc.index(test_matrix['KI %'][index])]
			nozzleD = test_matrix['Nozzle Diameter (um)'][index]
			KIperc = test_matrix['KI %'][index]
			TtoEPL = model[0] 
			EPLtoT = model[1]
			
			graph_folder = create_folder(project_folder + '/Processed/' + scintillator + '/Graphs/' + test_name)
			width_folder = create_folder(project_folder + '/Processed/' + scintillator + '/Widths/' + test_name)

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

			# Rotate the 700 um jet
			if '700' in test_name:
				data_epl = rotate(data_epl, 2.0)

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
				peaks, _ = find_peaks(smoothed, width=50, prominence=0.01)
				warnings.filterwarnings('default')
				
				if len(peaks) == 1:
					# Ellipse fitting
					warnings.filterwarnings('ignore')
					[relative_width, relative_max, lpos, rpos] = peak_widths(smoothed, peaks, rel_height=0.80)
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
						plot_ellipse(epl_graph[z], fitted_graph[z], graph_folder + '/' + scintillator + '_{0:03.0f}'.format(k) + '.png')
						plt.close()

						plot_widths(data_epl[k, :], peaks, relative_max, lpos[0], rpos[0], width_folder + '/' + scintillator + '_{0:03.0f}'.format(k) + '.png')
						plt.close()
			
			if len(peaks) == 1:
				signal = np.nanmean(data_epl[20:325, int(round(np.nanmean(lateral_position)))-20:int(round(np.nanmean(lateral_position)))+20])
			else:
				signal = 0
				
			noise = np.nanstd(data_epl[offset_sl_x, offset_sl_y])
			SNR = signal / noise
			
			try:
				idx = np.isfinite(ellipse_epl) & np.isfinite(optical_diameter)
				linfits, linfits_err, r2_values = lin_fit(np.array(ellipse_epl)[idx], np.array(optical_diameter)[idx])
				
			except:
				pass
			
			# Calculate ratios
			ratio_ellipseT = np.array(ellipse_T) / np.array(optical_T)
			ratio_peakT = np.array(peak_T) / np.array(optical_T)
			ratio_ellipse = np.array(ellipse_epl) / np.array(optical_diameter)
			ratio_peak = np.array(peak_epl) / np.array(optical_diameter)

			mean_ratio_ellipseT = np.nanmean(ratio_ellipseT)
			mean_ratio_peakT = np.nanmean(ratio_peakT)
			mean_ratio_ellipse = np.nanmean(ratio_ellipse)
			mean_ratio_peak = np.nanmean(ratio_peak)

			cv_ratio_ellipseT = np.nanstd(ratio_ellipseT) / np.nanmean(ratio_ellipseT)
			cv_ratio_peakT = np.nanstd(ratio_peakT) / np.nanmean(ratio_peakT)
			cv_ratio_ellipse = np.nanstd(ratio_ellipse) / np.nanmean(ratio_ellipse)
			cv_ratio_peak = np.nanstd(ratio_peak) / np.nanmean(ratio_peak)

			# Write ratios to file
			with open(project_folder + '/Processed/' + scintillator + '/summary.txt', 'a+') as filetowrite:
				filetowrite.write('{0}\t{1}\t{2:0.0f}\t{3:0.3f}\t{4:0.3f}\t{5:0.3f}\t{6:0.3f}\t{7:0.3f}\t{8:0.3f}\t{9:0.3f}\t{10:0.3f}\n'.format(nozzleD, KIperc, 
								  np.nanmean(lateral_position), mean_ratio_peak, cv_ratio_peak, mean_ratio_peakT, cv_ratio_peakT, 
								  mean_ratio_ellipse, cv_ratio_ellipse, mean_ratio_ellipseT, cv_ratio_ellipseT))

			ellipse_errors = [rmse(ellipse_epl, optical_diameter), mape(ellipse_epl, optical_diameter), zeta(ellipse_epl, optical_diameter), mdlq(ellipse_epl, optical_diameter)]
			peak_errors = [rmse(peak_epl, optical_diameter), mape(peak_epl, optical_diameter), zeta(peak_epl, optical_diameter), mdlq(peak_epl, optical_diameter)]

			processed_data = {'Diameters': [optical_diameter, ellipse_epl, peak_epl], 'Transmissions': [optical_T, ellipse_T, peak_T], 'Axial Position': axial_position,
								 'Lateral Position': lateral_position, 'Bounds': [left_bound, right_bound], 'Linear Fits': linfits, 'Linear Fits Error': linfits_err,
								 'R2': r2_values, 'Offset EPL': offset_epl, 'SNR': SNR, 'Ellipse Errors': ellipse_errors, 'Peak Errors': peak_errors,
								 'Transmission Ratios': [ratio_ellipseT, ratio_peakT], 'EPL Ratios': [ratio_peak, ratio_ellipse]}

			with open(summary_folder + '/' + scintillator + '_' + test_name + '.pckl', 'wb') as f:
				pickle.dump(processed_data, f)

			# Boundary plot
			plt.figure()
			plt.imshow(data_epl, vmin=0, vmax=0.25, zorder=1)
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
			plt.ylim([0.4, 2.15])
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(vertical_ellipse_folder, scintillator, test_name))
			plt.close()

			# Vertical peak variation
			plt.figure()
			plt.plot(axial_position, np.array(peak_epl)*10, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Peak Diameter (mm)')
			plt.ylim([0.4, 2.15])
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(vertical_peak_folder, scintillator, test_name))
			plt.close()

			# Vertical optical variation
			plt.figure()
			plt.plot(axial_position, np.array(optical_diameter)*10, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Optical Diameter (mm)')
			plt.ylim([0.4, 2.15])
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(vertical_optical_folder, scintillator, test_name))
			plt.close()

			# Vertical EPL ratio (ellipse)
			plt.figure()
			plt.plot(axial_position, ratio_ellipse, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Ellipse/Optical EPL Ratio')
			plt.ylim([0.4, 1.1])
			plt.savefig('{0}/{1}_{2}.png'.format(ratio_ellipse_folder, scintillator, test_name))
			plt.close()

			# Vertical EPL ratio (peak)
			plt.figure()
			plt.plot(axial_position, ratio_peak, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Peak/Optical EPL Ratio')
			plt.ylim([0.4, 1.1])
			plt.savefig('{0}/{1}_{2}.png'.format(ratio_peak_folder, scintillator, test_name))
			plt.close()

			# Vertical transmission ratio (ellipse)
			plt.figure()
			plt.plot(axial_position, ratio_ellipseT, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Ellipse/Optical Transmission Ratio')
			plt.ylim([0.8, 1.3])
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(ratio_ellipseT_folder, scintillator, test_name))
			plt.close()

			# Vertical transmission ratio (peak)
			plt.figure()
			plt.plot(axial_position, ratio_peakT, ' o')
			plt.xlabel('Axial Position (px)')
			plt.ylabel('Peak/Optical Transmission Ratio')
			plt.ylim([0.8, 1.3])
			plt.title(test_name)
			plt.savefig('{0}/{1}_{2}.png'.format(ratio_peakT_folder, scintillator, test_name))
			plt.close()

# Run this script
if __name__ == '__main__':
	main()