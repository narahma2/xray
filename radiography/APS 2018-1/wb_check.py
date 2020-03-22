# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:40:17 2019

@author: rahmann
"""

import sys
sys.path.append('E:/OneDrive - purdue.edu/Research/GitHub/coding/python')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from Spectra.spectrum_modeling import multi_angle as xop, xcom, xcom_reshape, density_KIinH2O, beer_lambert, visible_light, beer_lambert_unknown
from scipy.interpolate import CubicSpline

#%% Spectra model
model = ["water", "KI1p6", "KI3p4", "KI4p8", "KI8p0", "KI10p0", "KI11p1"]
atten_avg = len(model) * [None]
trans_avg = len(model) * [None]

for i, m in enumerate(model):
    # Load XOP spectra and XCOM inputs
    input_folder = "R:/APS 2018-1/Imaging/Spectra_Inputs"
    input_spectra = xop(input_folder + "/xsurface1.dat")
    
    air_atten = xcom(input_folder + "/air.txt")
    liquid_atten = xcom(input_folder + "/" + m + ".txt")
    scint_atten = xcom(input_folder + "/YAG.txt")
    
    # Reshape XCOM x-axis to match XOP
    air_atten = xcom_reshape(air_atten, input_spectra["Energy"])
    liquid_atten = xcom_reshape(liquid_atten, input_spectra["Energy"])
    scint_atten = xcom_reshape(scint_atten, input_spectra["Energy"])
    
    # Experimental setup
    # EPL in cm
    air_epl = 70
    scint_epl = 0.05
    spray_epl = np.linspace(0, 1, 1000)
    
    # Density in g/cm^3
    air_den = 0.001275
    scint_den =  4.55
    if m == "water":
        spray_den = 1.0                         # KI 0%
    if m == "KI1p6":
        spray_den = density_KIinH2O(1.6)        # KI 1.6%
    if m == "KI3p4":
        spray_den = density_KIinH2O(3.4)        # KI 3.4%
    if m == "KI4p8":
        spray_den = density_KIinH2O(4.8)        # KI 4.8%
    if m == "KI8p0":
        spray_den = density_KIinH2O(8)          # KI 8.0%
    if m == "KI10p0":
        spray_den = density_KIinH2O(10)         # KI 10.0%
    if m == "KI11p1":
        spray_den = density_KIinH2O(11.1)       # KI 11.1%
        
    # Apply Beer-Lambert law
    # Scintillator absorption & response
    scint_trans = [beer_lambert(incident, scint_atten["Attenuation"], scint_den, scint_epl) for incident in input_spectra["Power"]]
    scint_vl = visible_light(input_spectra["Power"], scint_trans)
    
    # Filtered spectrum for I_0
    air_filter = [beer_lambert(incident, air_atten["Attenuation"], air_den, air_epl) for incident in scint_vl]
    
    # Model of X-ray -> Experiment -> Spray -> Scintillator
    spray_spectra = [beer_lambert_unknown(incident, liquid_atten["Attenuation"], spray_den, spray_epl) for incident in air_filter]
    
    # Total intensity calculations
    # Flat field
    I0 = [x for x in air_filter]
    
    # Spray
    I = [x for x in spray_spectra]
    
    Transmission = [x1/x2 for (x1,x2) in zip(I,I0)]
    
    break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    