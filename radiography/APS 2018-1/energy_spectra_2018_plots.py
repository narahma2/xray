"""
-*- coding: utf-8 -*-
Creates energy spectra plots for the whitebeam 2018-1 model.

@Author: rahmann
@Date:   2020-04-20 12:27:25
@Last Modified by:   rahmann
@Last Modified time: 2020-04-20 12:27:25
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
from whitebeam_2018 import spectra_angles

# Location of APS 2018-1 data
project_folder = sys_folder + '/X-ray Radiography/APS 2018-1/'

# Load XOP spectra
input_folder = project_folder + '/Spectra_Inputs'
input_spectra = xop(input_folder + '/xsurface1.dat')

# Find the angles corresponding to the 2018-1 image vertical pixels
angles_mrad = spectra_angles()

# Create an interpolation object based on angle
# Passing in an angle in mrad will output an interpolated spectra (w/ XOP as reference) 
spectra_linfit = interp1d(input_spectra['Angle'], input_spectra['Power'], axis=0)

# Create an array containing spectra corresponding to each row of the 2018-1 images
spectra2D = spectra_linfit(angles_mrad)

# Grab the edge spectra
spectra_edge = spectra2D[25]

# Load NIST XCOM attenuation curves
air_atten = xcom(input_folder + '/air.txt', att_column=5)
Be_atten = xcom(input_folder + '/Be.txt', att_column=5)
YAG_atten = xcom(input_folder + '/YAG.txt', att_column=3)
LuAG_atten = xcom(input_folder + '/Al5Lu3O12.txt', att_column=3)

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

# Find the detected spectra through the scintillators (transmitted spectra) at middle
LuAG_detected = beer_lambert(spectra_middle, LuAG_atten['Attenuation'], LuAG_den, LuAG_epl)
YAG_detected = beer_lambert(spectra_middle, YAG_atten['Attenuation'], YAG_den, YAG_epl)

# Same as above but on edges
LuAG_detected_edge = beer_lambert(spectra_edge, LuAG_atten['Attenuation'], LuAG_den, LuAG_epl)
YAG_detected_edge = beer_lambert(spectra_edge, YAG_atten['Attenuation'], YAG_den, YAG_epl)

# Find the scintillator response at middle
LuAG_response = LuAG_detected / spectra_middle
YAG_response = YAG_detected / spectra_middle

# Same as above but on edges
LuAG_response_edge = LuAG_detected_edge / spectra_edge
YAG_response_edge = YAG_detected_edge / spectra_edge

## Filtered spectrum
# Middle spectrum
spectra_middle_filtered = beer_lambert(spectra_middle, air_atten['Attenuation'], air_den, air_epl)
spectra_middle_filtered = beer_lambert(spectra_middle_filtered, Be_atten['Attenuation'], Be_den, Be_epl)
# Use Ben's correction method
spectra_middle_filtered = beer_lambert(spectra_middle_filtered, air_atten['Attenuation'], air_den, 50)

# Edge spectrum
spectra_edge_filtered = beer_lambert(spectra_edge, air_atten['Attenuation'], air_den, air_epl)
spectra_edge_filtered = beer_lambert(spectra_edge_filtered, Be_atten['Attenuation'], Be_den, Be_epl)
# Use Ben's correction method
spectra_edge_filtered = beer_lambert(spectra_edge_filtered, air_atten['Attenuation'], air_den, 50)

## Filtered (detected)
# Middle spectrum
LuAG_detected_filtered = spectra_middle_filtered * LuAG_response
YAG_detected_filtered = spectra_middle_filtered * YAG_response

# Edge spectrum
LuAG_detected_edge_filtered = spectra_edge_filtered * LuAG_response
YAG_detected_edge_filtered = spectra_edge_filtered * YAG_response

## Plot figures
plot_folder = project_folder + '/Figures/Energy_Spectra'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Scintillator response
plt.figure()
plt.plot(input_spectra['Energy'] / 1000, LuAG_response, linewidth=1.5, color='cornflowerblue', label='LuAG')
plt.plot(input_spectra['Energy'] / 1000, LuAG_response_edge, linestyle='--', linewidth=2.5, color='cornflowerblue', label='LuAG (edge)')
plt.plot(input_spectra['Energy'] / 1000, YAG_response, linewidth=1.5, color='seagreen', label='YAG')
plt.plot(input_spectra['Energy'] / 1000, YAG_response_edge, linestyle='--', linewidth=2.5, color='seagreen', label='YAG (edge)')
plt.legend()
plt.xlabel('Energy (keV)')
plt.ylabel('Scintillator Response')
plt.savefig(plot_folder + '/scintillator_response.png')

# Spectra along middle
plt.figure()
plt.plot(input_spectra['Energy'] / 1000, 1000*spectra_middle / (input_spectra['Energy'] / 1000), linewidth=1.5, color='black', label='Incident')
plt.plot(input_spectra['Energy'] / 1000, 1000*spectra_middle_filtered / (input_spectra['Energy'] / 1000), linewidth=1.5, color='red', label='Filtered')
plt.plot(input_spectra['Energy'] / 1000, 1000*LuAG_detected_filtered / (input_spectra['Energy'] / 1000), linewidth=1.5, color='cornflowerblue', linestyle='--', label='LuAG (detected)')
plt.plot(input_spectra['Energy'] / 1000, 1000*YAG_detected_filtered / (input_spectra['Energy'] / 1000), linewidth=1.5, color='seagreen', linestyle='--', label='YAG (detected)')
plt.legend()
plt.xlabel('Energy (keV)')
plt.ylabel('Spectral Power (mW/keV)')
plt.savefig(plot_folder + '/middle_spectra.png')

# Spectra along middle
plt.figure()
plt.plot(input_spectra['Energy'] / 1000, 1000*spectra_edge / (input_spectra['Energy'] / 1000), linewidth=1.5, color='black', label='Incident')
plt.plot(input_spectra['Energy'] / 1000, 1000*spectra_edge_filtered / (input_spectra['Energy'] / 1000), linewidth=1.5, color='red', label='Filtered')
plt.plot(input_spectra['Energy'] / 1000, 1000*LuAG_detected_edge_filtered / (input_spectra['Energy'] / 1000), linewidth=1.5, color='cornflowerblue', linestyle='--', label='LuAG (detected)')
plt.plot(input_spectra['Energy'] / 1000, 1000*YAG_detected_edge_filtered / (input_spectra['Energy'] / 1000), linewidth=1.5, color='seagreen', linestyle='--', label='YAG (detected)')
plt.legend()
plt.xlabel('Energy (keV)')
plt.ylabel('Spectral Power (mW/keV)')
plt.savefig(plot_folder + '/edge_spectra.png')