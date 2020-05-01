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

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from Spectra.spectrum_modeling import multi_angle as xop, xcom, xcom_reshape, density_KIinH2O, beer_lambert, visible_light, beer_lambert_unknown
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from jet_processing import create_folder

def spectra_angles(flat='{0}/X-ray Radiography/APS 2018-1/Images/Uniform_Jets/Mean/AVG_Jet_flat2.tif'.format(sys_folder)):
	"""
	Finds the center of the X-ray beam.
	=============
	--VARIABLES--
	flat:          Path to the flat field .TIFF file.
	"""
	# Load in the flat field into an array
	flatfield = np.array(Image.open(flat))

	# Average the flat field horizontally
	flatfield_avg = np.mean(flatfield, axis=1)

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

	return angles_mrad, flatfield_avg


def averaged_plots(x_var, y_var, ylabel, xlabel, yscale, name, project_folder, scintillator):
	"""
	Creates plots of the averaged variables.
	=============
	--VARIABLES--
	x_var:          	Path to the flat field .TIFF file.
	y_var:				Y axis variable.
	ylabel:				Y axis label.
	xlabel:				X axis label.
	yscale:				Y scale ('log' or 'linear').
	name:				Save name for the plot.
	project_folder:		Location of project folder.
	scintillator:		Scintillator being used.
	"""

	averaged_folder = create_folder('{0}/Figures/Averaged_Figures'.format(project_folder))

	plt.figure()
	plt.plot(x_var[0], y_var[0], color='k', linewidth=2.0, label='Water')
	plt.plot(x_var[1], y_var[1], marker='x', markevery=50, linewidth=2.0, label='1.6% KI')
	plt.plot(x_var[2], y_var[2], linestyle='--', linewidth=2.0, label='3.4% KI')
	plt.plot(x_var[3], y_var[3], linestyle='-.', linewidth=2.0, label='4.8% KI')
	plt.plot(x_var[4], y_var[4], linestyle=':', linewidth=2.0, label='8.0% KI')
	plt.plot(x_var[5], y_var[5], marker='^', markevery=50, linewidth=2.0, label='10.0% KI')
	plt.plot(x_var[6], y_var[6], linestyle='--', linewidth=2.0, label='11.1% KI')
	plt.legend()
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.yscale(yscale)
	plt.savefig(averaged_folder + '/{0}_{1}.png'.format(scintillator, name))


def scintillator_response2D(spectra_linfit, scintillator_atten, scintillator_den, scintillator_epl):
	## Scintillator response
	# Use the middle spectrum (highest peak) for the scintillator response curve (assume indep. of spectral intensity)
	# Scintillator response curves were verified in a previous commit to be the same vertically (center = edge)
	spectra_middle = spectra_linfit(0)

	# Find the transmitted spectrum through the scintillators at middle location
	scintillator_transmitted = beer_lambert(spectra_middle, scintillator_atten['Attenuation'], scintillator_den, scintillator_epl)

	# Find the detected visible light emission from the scintillator (absorbed spectrum)
	scintillator_detected = visible_light(spectra_middle, scintillator_transmitted)

	# Find the scintillator response at middle
	scintillator_response = scintillator_detected / spectra_middle

	return scintillator_response


def filtered_spectra(input_folder, input_spectra, spectra, scintillator_response):
	# Load NIST XCOM attenuation curves
	air_atten = xcom(input_folder + '/air.txt', att_column=5)
	Be_atten = xcom(input_folder + '/Be.txt', att_column=5)

	# Reshape XCOM x-axis to match XOP
	air_atten = xcom_reshape(air_atten, input_spectra['Energy'])
	Be_atten = xcom_reshape(Be_atten, input_spectra['Energy'])

	## EPL in cm
	# Air EPL
	air_epl = 70

	# Be window EPL
	# See Alan's 'Spray Diagnostics at the Advanced Photon Source 7-BM Beamline' (3 Be windows)
	Be_epl = 0.075

	## Density in g/cm^3
	# Air density
	air_den = 0.001275

	# Beryllium density
	Be_den = 1.85

	# Apply air filter
	spectra_filtered = beer_lambert(spectra, air_atten['Attenuation'], air_den, air_epl)

	# Apply Be window filter
	spectra_filtered = beer_lambert(spectra_filtered, Be_atten['Attenuation'], Be_den, Be_epl)

	# Apply correction filter (air)
	# spectra_filtered = beer_lambert(spectra_filtered, air_atten['Attenuation'], air_den, 50)

	# Find detected spectra
	spectra_detected = spectra_filtered * scintillator_response

	return spectra_filtered, spectra_detected


def spray_model(input_folder, input_spectra, m, spray_epl, scintillator, scintillator_detected, I0, project_folder):
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
	spray_detected = [beer_lambert_unknown(incident, liquid_atten['Attenuation'], spray_den, spray_epl) for incident in scintillator_detected]

	# Spray
	I = [np.trapz(x, input_spectra['Energy']) for x in spray_detected]
	
	## LHS of Beer-Lambert Law
	Transmission = [x1/x2 for (x1, x2) in zip(I, I0)]
	
	## Cubic spline fitting of Transmission and spray_epl curves (needs to be reversed b/c of monotonically increasing
	## restriction on 'x', however this does not change the interpolation call)
	# Function that takes in transmission value (I/I0) and outputs EPL (cm)
	TtoEPL = [CubicSpline(vertical_pix[::-1], spray_epl[::-1]) for vertical_pix in Transmission]
	
	# Function that takes in EPL (cm) value and outputs transmission value (I/I0)
	EPLtoT = [CubicSpline(spray_epl, vertical_pix) for vertical_pix in Transmission]
	
	# Save model
	model_folder = create_folder('{0}/Model/'.format(project_folder))

	with open(model_folder + '/{0}_model_{1}.pckl'.format(m, scintillator), 'wb') as f:
		pickle.dump([TtoEPL, EPLtoT, spray_epl, Transmission], f)

	# Calculate average attenuation and transmission
	atten_avg = np.nanmean([-np.log(x)/spray_epl for x in Transmission], axis=0)
	trans_avg = np.nanmean(Transmission, axis=0)

	return atten_avg, trans_avg


def main():
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
	angles_mrad, _ = spectra_angles()

	# Create an interpolation object based on angle
	# Passing in an angle in mrad will output an interpolated spectra (w/ XOP as reference) 
	spectra_linfit = interp1d(input_spectra['Angle'], input_spectra['Power'], axis=0)
	
	# Create an array containing spectra corresponding to each row of the 2018-1 images
	spectra2D = spectra_linfit(angles_mrad)

	# Load NIST XCOM attenuation curves
	YAG_atten = xcom(input_folder + '/YAG.txt', att_column=3)
	LuAG_atten = xcom(input_folder + '/Al5Lu3O12.txt', att_column=3)

	# Reshape XCOM x-axis to match XOP
	YAG_atten = xcom_reshape(YAG_atten, input_spectra['Energy'])
	LuAG_atten = xcom_reshape(LuAG_atten, input_spectra['Energy'])

	# Scintillator EPL
	YAG_epl = 0.05      # 500 um
	LuAG_epl = 0.01     # 100 um

	# Spray EPL
	spray_epl = np.linspace(0.001, 0.82, 820)

	# Scintillator densities from Crytur <https://www.crytur.cz/materials/yagce/>
	YAG_den =  4.57
	LuAG_den = 6.73

	### Apply Beer-Lambert law
	## Scintillator response
	LuAG_response = scintillator_response2D(spectra_linfit, LuAG_atten, LuAG_den, LuAG_epl)
	YAG_response = scintillator_response2D(spectra_linfit, YAG_atten, YAG_den, YAG_epl)

	# Apply filters and find detected visible light emission
	LuAG = map(lambda x: filtered_spectra(input_folder, input_spectra, x, LuAG_response), spectra2D)
	LuAG = list(LuAG)
	LuAG_detected = np.swapaxes(LuAG, 0, 1)[1]

	YAG = map(lambda x: filtered_spectra(input_folder, input_spectra, x, YAG_response), spectra2D)
	YAG = list(YAG)
	YAG_detected = np.swapaxes(YAG, 0, 1)[1]

	## Total intensity calculations
	# Flat field
	I0_LuAG = [np.trapz(x, input_spectra['Energy']) for x in LuAG_detected]
	I0_YAG = [np.trapz(x, input_spectra['Energy']) for x in YAG_detected]

	for i, m in enumerate(model):
		[atten_avg_LuAG[i], trans_avg_LuAG[i]] = spray_model(input_folder, input_spectra, m, spray_epl, 'LuAG', LuAG_detected, I0_LuAG, project_folder)
		[atten_avg_YAG[i], trans_avg_YAG[i]] = spray_model(input_folder, input_spectra, m, spray_epl, 'YAG', YAG_detected, I0_YAG, project_folder)
		
	with open('{0}/Model/averaged_variables_LuAG.pckl'.format(project_folder), 'wb') as f:
		pickle.dump([spray_epl, atten_avg_LuAG, trans_avg_LuAG], f)

	with open('{0}/Model/averaged_variables_YAG.pckl'.format(project_folder), 'wb') as f:
		pickle.dump([spray_epl, atten_avg_YAG, trans_avg_YAG], f)

	#%% Plot
	atten_avg = [atten_avg_LuAG, atten_avg_YAG]
	trans_avg = [trans_avg_LuAG, trans_avg_YAG]

	for i, scintillator in enumerate(['LuAG', 'YAG']):
		averaged_plots(trans_avg[i], atten_avg[i], 'Beam Avg. Atten. Coeff. [1/cm]', 'Transmission', 'log', 'coeff_vs_trans', project_folder, scintillator)
		averaged_plots(np.tile(10*np.array(spray_epl), [7, 1]), atten_avg[i], 'Beam Avg. Atten. Coeff. [1/cm]', 'EPL [mm]', 'log', 'coeff_vs_epl', project_folder, scintillator)
		averaged_plots(np.tile(10*np.array(spray_epl), [7, 1]), 1-np.array(trans_avg[i]), 'Attenuation', 'EPL [mm]', 'linear', 'atten_vs_epl', project_folder, scintillator)
		averaged_plots(np.tile(10*np.array(spray_epl), [7, 1]), trans_avg[i], 'Transmission', 'EPL [mm]', 'linear', 'trans_vs_epl', project_folder, scintillator)


# Run this script
if __name__ == '__main__':
	main()