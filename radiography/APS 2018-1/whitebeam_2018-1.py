# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:44:11 2019

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
    
    air_atten = xcom(input_folder + "/air.txt", att_column=5)
    liquid_atten = xcom(input_folder + "/" + m + ".txt", att_column=5)
    scint_atten = xcom(input_folder + "/YAG.txt", att_column=3)
    
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
    I0 = [np.trapz(x, input_spectra["Energy"]) for x in air_filter]
    
    # Spray
    I = [np.trapz(x, input_spectra["Energy"]) for x in spray_spectra]
    
    # LHS of Beer-Lambert Law
    Transmission = [x1/x2 for (x1,x2) in zip(I,I0)]
    
    # Cubic spline fitting of Transmission vs. spray_epl curves (needs to be reversed b/c of monotonically increasing
    # restriction on 'x', however this does not change the interpolation call)
    cs = [CubicSpline(angle[::-1], spray_epl[::-1]) for angle in Transmission]
    
    f = open("R:/APS 2018-1/Imaging/" + m + "_model.pckl", "wb")
    pickle.dump([input_spectra["Angle"], cs, spray_epl, Transmission], f)
    f.close()
    
    atten_avg[i] = np.nanmean([-np.log(x)/spray_epl for x in Transmission], axis=0)
    trans_avg[i] = np.nanmean(Transmission, axis=0)
    
f = open("R:/APS 2018-1/Imaging/averaged_variables.pckl", "wb")
pickle.dump([spray_epl, atten_avg, trans_avg], f)
f.close()

#%% Plot
plt.figure()
plt.plot(trans_avg[0], atten_avg[0], color='k', linewidth=2.0, label='Water')
plt.plot(trans_avg[1], atten_avg[1], marker='x', markevery=50, linewidth=2.0, label='1.6% KI')
plt.plot(trans_avg[2], atten_avg[2], linestyle='--', linewidth=2.0, label='3.4% KI')
plt.plot(trans_avg[3], atten_avg[3], linestyle='-.', linewidth=2.0, label='4.8% KI')
plt.plot(trans_avg[4], atten_avg[4], linestyle=':', linewidth=2.0, label='8.0% KI')
plt.plot(trans_avg[5], atten_avg[5], marker='^', markevery=50, linewidth=2.0, label='10.0% KI')
plt.plot(trans_avg[6], atten_avg[6], linestyle='--', linewidth=2.0, label='11.1% KI')
plt.legend()
plt.ylabel('Beam Avg. Atten. Coeff. [1/cm]')
plt.xlabel('Transmission')
plt.yscale('log')
plt.ylim([0.05, 10.95])
plt.xlim([0, 1])
plt.savefig('R:\\APS 2018-1\\Imaging\\WhiteBeam_Figures\\coeff_vs_trans.png')

plt.figure()
plt.plot(10*np.array(spray_epl), atten_avg[0], color='k', linewidth=2.0, label='Water')
plt.plot(10*np.array(spray_epl), atten_avg[1], marker='x', markevery=50, linewidth=2.0, label='1.6% KI')
plt.plot(10*np.array(spray_epl), atten_avg[2], linestyle='--', linewidth=2.0, label='3.4% KI')
plt.plot(10*np.array(spray_epl), atten_avg[3], linestyle='-.', linewidth=2.0, label='4.8% KI')
plt.plot(10*np.array(spray_epl), atten_avg[4], linestyle=':', linewidth=2.0, label='8.0% KI')
plt.plot(10*np.array(spray_epl), atten_avg[5], marker='^', markevery=50, linewidth=2.0, label='10.0% KI')
plt.plot(10*np.array(spray_epl), atten_avg[6], linestyle='--', linewidth=2.0, label='11.1% KI')
plt.legend()
plt.ylabel('Beam Avg. Atten. Coeff. [1/cm]')
plt.xlabel('EPL [mm]')
plt.yscale('log')
plt.ylim([0.05, 10.95])
plt.savefig('R:\\APS 2018-1\\Imaging\\WhiteBeam_Figures\\coeff_vs_epl.png')

plt.figure()
plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[0]), color='k', linewidth=2.0, label='Water')
plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[1]), marker='x', markevery=50, linewidth=2.0, label='1.6% KI')
plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[2]), linestyle='--', linewidth=2.0, label='3.4% KI')
plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[3]), linestyle='-.', linewidth=2.0, label='4.8% KI')
plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[4]), linestyle=':', linewidth=2.0, label='8.0% KI')
plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[5]), marker='^', markevery=50, linewidth=2.0, label='10.0% KI')
plt.plot(10*np.array(spray_epl), 1-np.array(trans_avg[6]), linestyle='--', linewidth=2.0, label='11.1% KI')
plt.legend()
plt.ylabel('Attenuation')
plt.xlabel('EPL [mm]')
plt.ylim([0, 1])
plt.savefig('R:\\APS 2018-1\\Imaging\\WhiteBeam_Figures\\atten_vs_epl.png')