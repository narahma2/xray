# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:22:00 2018

@author: rahmann
"""
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
from scipy.signal import find_peaks
from Spectra.spectrum_modeling import multi_angle as xop, xcom, xcom_reshape, beer_lambert, visible_light, beer_lambert_unknown

# Load XOP spectra and XCOM inputs
input_folder = "D:/Naveed/X-ray Chris/Spectrum_Input"
input_spectra = xop(input_folder + "/xop_power_2018APS.dat")

be_atten = xcom(input_folder + "/beryllium.txt")
kapton_atten = xcom(input_folder + "/kapton.txt")
air_atten = xcom(input_folder + "/air.txt")
ki10_atten = xcom(input_folder + "/KI10.txt")
ki50_atten = xcom(input_folder + "/KI50.txt")
scint_atten = xcom(input_folder + "/Al5Lu3O12.txt")

# Reshape XCOM x-axis to match XOP
be_atten = xcom_reshape(be_atten, input_spectra["Energy"])
kapton_atten = xcom_reshape(kapton_atten, input_spectra["Energy"])
air_atten = xcom_reshape(air_atten, input_spectra["Energy"])
ki10_atten = xcom_reshape(ki10_atten, input_spectra["Energy"])
ki50_atten = xcom_reshape(ki50_atten, input_spectra["Energy"])
scint_atten = xcom_reshape(scint_atten, input_spectra["Energy"])

# Experimental setup
# EPL in cm
be_epl = 0.075
kapton_epl = 0.01
air_epl = 70
scint_epl = 0.01
spray_epl = np.linspace(0, 2, 200)

# Density in g/cm^3
be_den = 1.848
kapton_den = 1.42
air_den = 0.001275
scint_den = 6.71
spray_den = 1.0731  # KI 10%

# Apply Beer-Lambert law
# Model of X-ray -> Scintillator
scint_trans = [beer_lambert(incident, scint_atten["Attenuation"], scint_den, scint_epl) for incident in input_spectra["Power"]]
scint_vl = visible_light(input_spectra["Power"], scint_trans)

# Model of X-ray -> Experiment -> Scintillator
be_trans = [beer_lambert(incident, be_atten["Attenuation"], be_den, be_epl) for incident in input_spectra["Power"]]
kapton_trans = [beer_lambert(incident, kapton_atten["Attenuation"], kapton_den, kapton_epl) for incident in be_trans]
air_trans = [beer_lambert(incident, air_atten["Attenuation"], air_den, air_epl) for incident in kapton_trans]
exp_scint_trans = [beer_lambert(incident, scint_atten["Attenuation"], scint_den, scint_epl) for incident in air_trans]
exp_scint_vl = visible_light(air_trans, exp_scint_trans)

# Model of X-ray -> Experiment -> Spray -> Scintillator
spray_trans = [beer_lambert_unknown(incident, ki10_atten["Attenuation"], spray_den, spray_epl) for incident in air_trans]
spray_scint_trans = [beer_lambert(incident, scint_atten["Attenuation"], scint_den, scint_epl) for incident in spray_trans]
spray_scint_vl = visible_light(spray_trans, spray_scint_trans)

# Total intensity calculations
# Flat field
I0 = [np.trapz(x, input_spectra["Energy"]) for x in exp_scint_vl]

# Spray
I = [np.trapz(x, input_spectra["Energy"]) for x in spray_scint_vl]

# LHS of Beer-Lambert Law
Transmission = [x1/x2 for (x1,x2) in zip(I,I0)]

# Polynomial fitting of spray_epl vs. Transmssion curves
z = [np.polyfit(angle, spray_epl, 13) for angle in Transmission]
p = [np.poly1d(angle) for angle in z]

# 700 um jet check
j = -1
binary_path = "D:/Naveed/X-ray Chris/2018APS_Check/Binary/"
widths = []
epl_max = []

for filename in os.listdir("D:/Naveed/X-ray Chris/2018APS_Check/J700_KI10_P0_Fn"):
    j += 1
    
    binary700 = tiff.imread(binary_path + os.listdir(binary_path)[j])
    wd = [find_peaks(row, width=50)[1]['widths'] for row in binary700]
    widths.append(wd[0:340])
    lft = [find_peaks(row, width=50)[1]['left_bases'] for row in binary700]
    rht = [find_peaks(row, width=50)[1]['right_bases'] for row in binary700]
    
    epl700 = np.empty([352, 768], dtype=float)
    check700 = tiff.imread("D:/Naveed/X-ray Chris/2018APS_Check/J700_KI10_P0_Fn/" + filename)
    check700[check700 > 1] = 1
    epl = []
    
    for i in range(len(check700)):
        epl700[i, :] = p[i](check700[i, :])
        if i < 340:
            epl.append(np.max(epl700[i, lft[i][0]:rht[i][0]]))
    
    epl_max.append(epl)
    epl700[epl700 < 0] = 0
    tiff.imsave("D:/Naveed/X-ray Chris/2018APS_Check/J700_KI10_P0_Fn_EPL/" + filename, np.float32(epl700))

#plt.figure()
#plt.plot(input_spectra["Energy"], input_spectra["Power"][168], 'b,--')
#plt.plot(input_spectra["Energy"], scint_vl[168], 'b')
#plt.plot(input_spectra["Energy"], input_spectra["Power"][0], 'k,--')
#plt.plot(input_spectra["Energy"], scint_vl[0], 'k')