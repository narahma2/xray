# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:44:11 2019

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

import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from Spectra.spectrum_modeling import multi_angle as xop, xcom, xcom_reshape, density_KIinH2O, beer_lambert, visible_light, beer_lambert_unknown
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def spectra_angles(flat='{0}/X-ray Radiography/APS 2018-1/Images/Uniform_Jets/Mean/AVG_Jet_flat2.tif'.format(sys_folder)):
	"""
	Finds the center of the X-ray beam.
	=============
	--VARIABLES--
	flat:          Path to the flat field .TIFF file.
	"""
	# Load in the flat field into an array
	flatfield = np.array(Image.open(flat))

	# Initialize the array to find the vertical middle of the beam
	beam_middle = np.zeros((flatfield.shape[1]))

	# Loop each column of the image to find the vertical middle
	for i in range(flatfield.shape[1]):
		warnings.filterwarnings('ignore')
		beam_middle[i] = np.argmax(savgol_filter(flatfield[:,i], 55, 3))
		warnings.filterwarnings('default')

	# Take the mean to be the center
	beam_middle_avg = int(np.mean(beam_middle).round())

	# Image pixel size
	cm_pix = 0.16 / 162

	# Distance X-ray beam travels
	length = 3500

	# Convert image vertical locations to angles
	vertical_indices = np.linspace(0, flatfield.shape[0], flatfield.shape[0])
	vertical_indices -= beam_middle_avg
	angles_mrad = np.arctan((vertical_indices*cm_pix) / length) * 1000

	return angles_mrad

if __name__ == '__main__':
	# Location of APS 2018-1 data
	project_folder = '{0}/X-ray Radiography/APS 2018-1/'.format(sys_folder)

	model = ['water', 'KI1p6', 'KI3p4', 'KI4p8', 'KI8p0', 'KI10p0', 'KI11p1']
	atten_avg_LuAG = len(model) * [None]
	trans_avg_LuAG = len(model) * [None]
	atten_avg_YAG = len(model) * [None]
	trans_avg_YAG = len(model) * [None]

	# Load XOP spectra
	input_folder = '{0}/Spectra_Inputs'.format(project_folder)
	input_spectra = xop('{0}/xsurface1.dat'.format(input_folder))

	# Find the angles corresponding to the 2018-1 image vertical pixels
	angles_mrad = spectra_angles()

	# Create an interpolation object based on angle
	# Passing in an angle in mrad will output an interpolated spectra (w/ XOP as reference) 
	spectra_linfit = interp1d(input_spectra['Angle'], input_spectra['Power'], axis=0)

	# Create an array containing spectra corresponding to each row of the 2018-1 images
	spectra2D = spectra_linfit(angles_mrad)

	# Load NIST XCOM attenuation curves
	air_atten = xcom('{0}/air.txt'.format(input_folder), att_column=5)
	Be_atten = xcom('{0}/Be.txt'.format(input_folder), att_column=5)
	YAG_atten = xcom('{0}/YAG.txt'.format(input_folder), att_column=3)
	LuAG_atten = xcom('{0}/Al5Lu3O12.txt'.format(input_folder), att_column=3)

	# Reshape XCOM x-axis to match XOP
	air_atten = xcom_reshape(air_atten, input_spectra['Energy'])
	Be_atten = xcom_reshape(Be_atten, input_spectra['Energy'])
	YAG_atten = xcom_reshape(YAG_atten, input_spectra['Energy'])
	LuAG_atten = xcom_reshape(LuAG_atten, input_spectra['Energy'])

	## EPL in cm
	# Air EPL
	air_epl = 70

	# Be window EPL
	# See Alan's 'Spray Diagnostics at the Advanced Photon Source 7-BM Beamline' (3 Be windows)
	Be_epl = 0.075

	# Scintillator EPL
	YAG_epl = 0.05      # 500 um
	LuAG_epl = 0.01     # 100 um

	# Spray EPL
	spray_epl = np.linspace(0, 1, 1000)

	## Density in g/cm^3
	# Air density
	air_den = 0.001275

	# Beryllium density
	Be_den = 1.85

	# Scintillator densities from Crytur <https://www.crytur.cz/materials/yagce/>
	YAG_den =  4.57
	LuAG_den = 6.73

	### Apply Beer-Lambert law
	## Scintillator response
	# Use the middle spectrum (highest peak) for the scintillator response curve (assume indep. of spectral intensity)
	spectra_middle = spectra_linfit(0)

	# Find the detected spectra through the scintillators (transmitted spectra)
	LuAG_detected = beer_lambert(spectra_middle, LuAG_atten['Attenuation'], LuAG_den, LuAG_epl)
	YAG_detected = beer_lambert(spectra_middle, YAG_atten['Attenuation'], YAG_den, YAG_epl)

	# Find the scintillator response
	LuAG_response = LuAG_detected / spectra_middle
	YAG_response = YAG_detected / spectra_middle

	## Visible spectra (no spray)
	visible_LuAG = spectra2D * LuAG_response
	visible_YAG = spectra2D * YAG_response

	## Hardened visible spectra
	# Air
	visible_LuAG_hardened = [beer_lambert(incident, air_atten['Attenuation'], air_den, air_epl) for incident in visible_LuAG]
	visible_YAG_hardened = [beer_lambert(incident, air_atten['Attenuation'], air_den, air_epl) for incident in visible_YAG]

	# Be windows
	visible_LuAG_hardened = [beer_lambert(incident, Be_atten['Attenuation'], Be_den, Be_epl) for incident in visible_LuAG_hardened]
	visible_YAG_hardened = [beer_lambert(incident, Be_atten['Attenuation'], Be_den, Be_epl) for incident in visible_YAG_hardened]

	# Correction filter (using air, as Ben did)
	# visible_LuAG_hardened = [beer_lambert(incident, air_atten['Attenuation'], air_den, 50) for incident in visible_LuAG_hardened]
	# visible_YAG_hardened = [beer_lambert(incident, air_atten['Attenuation'], air_den, 50) for incident in visible_YAG_hardened]

	## Total intensity calculations
	# Flat field
	I0_LuAG = [np.trapz(x, input_spectra['Energy']) for x in visible_LuAG_hardened]
	I0_YAG = [np.trapz(x, input_spectra['Energy']) for x in visible_YAG_hardened]

	for i, m in enumerate(model):
		# Spray attenuation curves
		liquid_atten = xcom('{0}/{1}.txt'.format(input_folder, m), att_column=5)
		liquid_atten = xcom_reshape(liquid_atten, input_spectra['Energy'])

		# Spray densities
		if m == 'water':
			spray_den = 1.0                         # KI 0%
		if m == 'KI1p6':
			spray_den = density_KIinH2O(1.6)        # KI 1.6%
		if m == 'KI3p4':
			spray_den = density_KIinH2O(3.4)        # KI 3.4%
		if m == 'KI4p8':
			spray_den = density_KIinH2O(4.8)        # KI 4.8%
		if m == 'KI8p0':
			spray_den = density_KIinH2O(8)          # KI 8.0%
		if m == 'KI10p0':
			spray_den = density_KIinH2O(10)         # KI 10.0%
		if m == 'KI11p1':
			spray_den = density_KIinH2O(11.1)       # KI 11.1%
		
		## Add in the spray
		spray_LuAG = [beer_lambert_unknown(incident, liquid_atten['Attenuation'], spray_den, spray_epl) for incident in visible_LuAG_hardened]
		spray_YAG = [beer_lambert_unknown(incident, liquid_atten['Attenuation'], spray_den, spray_epl) for incident in visible_YAG_hardened]

		# Spray
		I_LuAG = [np.trapz(x, input_spectra['Energy']) for x in spray_LuAG]
		I_YAG = [np.trapz(x, input_spectra['Energy']) for x in spray_YAG]
		
		## LHS of Beer-Lambert Law
		Transmission_LuAG = [x1/x2 for (x1, x2) in zip(I_LuAG, I0_LuAG)]
		Transmission_YAG = [x1/x2 for (x1, x2) in zip(I_YAG, I0_YAG)]
		
		# Cubic spline fitting of Transmission vs. spray_epl curves (needs to be reversed b/c of monotonically increasing
		# restriction on 'x', however this does not change the interpolation call)
		cs_LuAG = [CubicSpline(vertical_px[::-1], spray_epl[::-1]) for vertical_px in Transmission_LuAG]
		cs_YAG = [CubicSpline(vertical_px[::-1], spray_epl[::-1]) for vertical_px in Transmission_YAG]
		
		with open('{0}/Model/{1}_model_LuAG.pckl'.format(project_folder, m), 'wb') as f:
			pickle.dump([cs_LuAG, spray_epl, Transmission_LuAG], f)

		with open('{0}/Model/{1}_model_YAG.pckl'.format(project_folder, m), 'wb') as f:
			pickle.dump([cs_YAG, spray_epl, Transmission_YAG], f)
		
		atten_avg_LuAG[i] = np.nanmean([-np.log(x)/spray_epl for x in Transmission_LuAG], axis=0)
		trans_avg_LuAG[i] = np.nanmean(Transmission_LuAG, axis=0)

		atten_avg_YAG[i] = np.nanmean([-np.log(x)/spray_epl for x in Transmission_YAG], axis=0)
		trans_avg_YAG[i] = np.nanmean(Transmission_YAG, axis=0)
		
	with open('{0}/Model/averaged_variables_LuAG.pckl'.format(project_folder), 'wb') as f:
		pickle.dump([spray_epl, atten_avg_LuAG, trans_avg_LuAG], f)

	with open('{0}/Model/averaged_variables_YAG.pckl'.format(project_folder), 'wb') as f:
		pickle.dump([spray_epl, atten_avg_YAG, trans_avg_YAG], f)

	#%% Plot
	atten_avg = [atten_avg_LuAG, atten_avg_YAG]
	trans_avg = [trans_avg_LuAG, trans_avg_YAG]

	for i, scintillator in enumerate(['LuAG', 'YAG']):
		plt.figure()
		plt.plot(trans_avg[i][0], atten_avg[i][0], color='k', linewidth=2.0, label='Water')
		plt.plot(trans_avg[i][1], atten_avg[i][1], marker='x', markevery=50, linewidth=2.0, label='1.6% KI')
		plt.plot(trans_avg[i][2], atten_avg[i][2], linestyle='--', linewidth=2.0, label='3.4% KI')
		plt.plot(trans_avg[i][3], atten_avg[i][3], linestyle='-.', linewidth=2.0, label='4.8% KI')
		plt.plot(trans_avg[i][4], atten_avg[i][4], linestyle=':', linewidth=2.0, label='8.0% KI')
		plt.plot(trans_avg[i][5], atten_avg[i][5], marker='^', markevery=50, linewidth=2.0, label='10.0% KI')
		plt.plot(trans_avg[i][6], atten_avg[i][6], linestyle='--', linewidth=2.0, label='11.1% KI')
		plt.legend()
		plt.ylabel('Beam Avg. Atten. Coeff. [1/cm]')
		plt.xlabel('Transmission')
		plt.yscale('log')
		# plt.ylim([0.05, 10.95])
		plt.xlim([0, 1])
		plt.savefig('{0}/Figures/Averaged_Figures/{1}_coeff_vs_trans.png'.format(project_folder, scintillator))

		plt.figure()
		plt.plot(10*np.array(spray_epl), atten_avg[i][0], color='k', linewidth=2.0, label='Water')
		plt.plot(10*np.array(spray_epl), atten_avg[i][1], marker='x', markevery=50, linewidth=2.0, label='1.6% KI')
		plt.plot(10*np.array(spray_epl), atten_avg[i][2], linestyle='--', linewidth=2.0, label='3.4% KI')
		plt.plot(10*np.array(spray_epl), atten_avg[i][3], linestyle='-.', linewidth=2.0, label='4.8% KI')
		plt.plot(10*np.array(spray_epl), atten_avg[i][4], linestyle=':', linewidth=2.0, label='8.0% KI')
		plt.plot(10*np.array(spray_epl), atten_avg[i][5], marker='^', markevery=50, linewidth=2.0, label='10.0% KI')
		plt.plot(10*np.array(spray_epl), atten_avg[i][6], linestyle='--', linewidth=2.0, label='11.1% KI')
		plt.legend()
		plt.ylabel('Beam Avg. Atten. Coeff. [1/cm]')
		plt.xlabel('EPL [mm]')
		plt.yscale('log')
		# plt.ylim([0.05, 10.95])
		plt.savefig('{0}/Figures/Averaged_Figures/{1}_coeff_vs_epl.png'.format(project_folder, scintillator))

		plt.figure()
		plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[i][0]), color='k', linewidth=2.0, label='Water')
		plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[i][1]), marker='x', markevery=50, linewidth=2.0, label='1.6% KI')
		plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[i][2]), linestyle='--', linewidth=2.0, label='3.4% KI')
		plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[i][3]), linestyle='-.', linewidth=2.0, label='4.8% KI')
		plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[i][4]), linestyle=':', linewidth=2.0, label='8.0% KI')
		plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[i][5]), marker='^', markevery=50, linewidth=2.0, label='10.0% KI')
		plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[i][6]), linestyle='--', linewidth=2.0, label='11.1% KI')
		plt.legend()
		plt.ylabel('Attenuation')
		plt.xlabel('EPL [mm]')
		plt.ylim([0, 1])
		plt.savefig('{0}/Figures/Averaged_Figures/{1}_atten_vs_epl.png'.format(project_folder, scintillator))

		plt.figure()
		plt.plot(10*np.array(spray_epl), np.array(trans_avg[i][0]), color='k', linewidth=2.0, label='Water')
		plt.plot(10*np.array(spray_epl), np.array(trans_avg[i][1]), marker='x', markevery=50, linewidth=2.0, label='1.6% KI')
		plt.plot(10*np.array(spray_epl), np.array(trans_avg[i][2]), linestyle='--', linewidth=2.0, label='3.4% KI')
		plt.plot(10*np.array(spray_epl), np.array(trans_avg[i][3]), linestyle='-.', linewidth=2.0, label='4.8% KI')
		plt.plot(10*np.array(spray_epl), np.array(trans_avg[i][4]), linestyle=':', linewidth=2.0, label='8.0% KI')
		plt.plot(10*np.array(spray_epl), np.array(trans_avg[i][5]), marker='^', markevery=50, linewidth=2.0, label='10.0% KI')
		plt.plot(10*np.array(spray_epl), np.array(trans_avg[i][6]), linestyle='--', linewidth=2.0, label='11.1% KI')
		plt.legend()
		plt.ylabel('Transmission')
		plt.xlabel('EPL [mm]')
		plt.ylim([0, 1])
		plt.savefig('{0}/Figures/Averaged_Figures/{1}_trans_vs_epl.png'.format(project_folder, scintillator))