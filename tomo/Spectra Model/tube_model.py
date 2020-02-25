# -*- coding: utf-8 -*-
"""
Creates the tube source spectra based on XOP.

Spectra Modeling Workflow
-------------------------
tube_model -> TREX_calibration -> TREX_correctionfactor -> TREX_error -> TREX_summary


Created on Fri Sep  6 15:37:01 2019

@author: rahmann
"""

import sys
sys.path.append('E:/General Scripts/python')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from Spectra.spectrum_modeling import tube_xop as import_tube, xcom, xcom_reshape, density_KIinH2O, beer_lambert, visible_light, beer_lambert_unknown

#%% Spectra model
model = ["Water", "50p0_KI"]
atten_coeff = len(model) * [None]
Transmission = len(model) * [None]

for i, m in enumerate(model):
    # Load XOP spectra and XCOM inputs
    input_folder = "R:/X-ray Tomography/Spectra Model"
    input_spectra = import_tube(input_folder + "/Modeling/80kV_flux.txt")
    
    # Attenuation coefficient in cm^2/g, convert to mm^2/g
    air_atten = xcom(input_folder + "/Modeling/Air.txt", att_column=5)
    air_atten["Attenuation"] *= 100
    air_atten["Attenuation"].name = 'Total Attenuation (mm^2/g) Without Coherent Scattering'
    
    al_atten = xcom(input_folder + "/Modeling/Al.txt", att_column=5)
    al_atten["Attenuation"] *= 100
    al_atten["Attenuation"].name = 'Total Attenuation (mm^2/g) Without Coherent Scattering'
    
    liquid_atten = xcom(input_folder + "/Modeling/" + m + ".txt", att_column=5)
    liquid_atten["Attenuation"] *= 100
    liquid_atten["Attenuation"].name = 'Total Attenuation (mm^2/g) Without Coherent Scattering'
    
    scint_atten = xcom(input_folder + "/Modeling/CsI.txt", att_column=3)
    scint_atten["Attenuation"] *= 100
    scint_atten["Attenuation"].name = 'Photoelectric Absorption (mm^2/g)'
    
    # Reshape XCOM x-axis to match XOP
    air_atten = xcom_reshape(air_atten, input_spectra["Energy"])
    al_atten = xcom_reshape(al_atten, input_spectra["Energy"])
    liquid_atten = xcom_reshape(liquid_atten, input_spectra["Energy"])
    scint_atten = xcom_reshape(scint_atten, input_spectra["Energy"])
    
    # Experimental setup
    # Units in mm
    air_epl = 340
    al_epl = 0.5
    spray_epl = np.linspace(0, 10, 1000)
    scint_epl = 0.15
    
    # Density in g/mm^3
    air_den = 1.275E-6
    al_den = 0.00271
    scint_den =  0.00451
    if m == "water":
        spray_den = 0.001                           # KI 0%
    else:
        spray_den = density_KIinH2O(50) / 1000      # KI 50%
        
    # Apply Beer-Lambert law
    # Scintillator absorption & response
    scint_trans = beer_lambert(input_spectra["Power"], scint_atten["Attenuation"], scint_den, scint_epl)
    scint_vl = visible_light(input_spectra["Power"], scint_trans)
    
    # Filtered spectrum for I_0
    al_window = beer_lambert(scint_vl, al_atten["Attenuation"], al_den, al_epl)
    air_filter = beer_lambert(al_window, air_atten["Attenuation"], air_den, air_epl)
    
    # Model of X-ray -> Experiment -> Spray -> Scintillator
    spray_spectra = beer_lambert_unknown(air_filter, liquid_atten["Attenuation"], spray_den, spray_epl)
    
    # Total intensity calculations
    # Flat field
    I0 = np.trapz(air_filter, input_spectra["Energy"])
    
    # Spray
    I = np.trapz(spray_spectra, input_spectra["Energy"])
    
    # LHS of Beer-Lambert Law
    Transmission[i] = I/I0
    
    # Cubic spline fitting of Transmission vs. spray_epl curves (needs to be reversed b/c of monotonically increasing
    # restriction on 'x', however this does not change the interpolation call)
    cs = CubicSpline(Transmission[i][::-1], spray_epl[::-1])
    
    f = open(input_folder + "/Modeling/" + m + "_model.pckl", "wb")
    pickle.dump([cs, spray_epl, Transmission[i]], f)
    f.close()

    # Overall experimental attenuation coefficient in 1/mm
    atten_coeff[i] = -np.log(Transmission[i])/spray_epl
    
#%% Plot
plt.figure()
plt.plot(Transmission[0], atten_coeff[0]*10, color='k', linewidth=2.0, label='Water')
plt.plot(Transmission[1], atten_coeff[1]*10, marker='x', markevery=50, linewidth=2.0, label='50% KI')
plt.legend()
plt.ylabel('Overall Model Atten. Coeff. [1/cm]')
plt.xlabel('Transmission')
#plt.yscale('log')
plt.ylim([0.05, 10.95])
plt.xlim([0, 1])
plt.savefig(input_folder + '\\Figures\\coeff_vs_trans.png')

plt.figure()
plt.plot(np.array(spray_epl), atten_coeff[0]*10, color='k', linewidth=2.0, label='Water')
plt.plot(np.array(spray_epl), atten_coeff[1]*10, marker='x', markevery=50, linewidth=2.0, label='50% KI')
plt.legend()
plt.ylabel('Overall Model Atten. Coeff. [1/cm]')
plt.xlabel('EPL [mm]')
#plt.yscale('log')
plt.ylim([0.05, 10.95])
plt.savefig(input_folder + '\\Figures\\coeff_vs_epl.png')

plt.figure()
plt.plot(np.array(spray_epl), 1-np.array(Transmission[0]), color='k', linewidth=2.0, label='Water')
plt.plot(np.array(spray_epl), 1-np.array(Transmission[1]), marker='x', markevery=50, linewidth=2.0, label='50% KI')
plt.legend()
plt.ylabel('Attenuation')
plt.xlabel('EPL [mm]')
plt.ylim([0, 1])
plt.savefig(input_folder + '\\Figures\\atten_vs_epl.png')