# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:04:07 2018

@author: Naveed
"""
import sys
sys.path.append('./')

import numpy as np
from spectra_functions import openXCOM
from scipy.interpolate import interp1d

#%% SETUP
V = 80                              # X-ray tube input voltage
    
keV = np.arange(1,200.1,0.1)        # Photon energy range

root = 'E:/Edwards Nozzle'
XOP_data = 'E:/Edwards Nozzle/XOP Data'

#%% REFERENCE DATA: Cross Sections and Densities XOP / NIST
c=3     # Photoelectric Absorption

acs_CsI = openXCOM(XOP_data + '/XOP-2p4_CsI.txt', c)        # Absorption Cross Section [cm^2/g]
mu_CsI = 4.51 * acs_CsI     # Abs Coeff = ACS * Density [ 1/cm = cm^2/g * g/cm^3 ]

c=7     # Total Attenuation
acs_Air =  openXCOM(XOP_data + '/XOP-2p4_Air.txt',c)          # Absorption Cross Section [cm^2/g]
mu_Air = 0.001225 * acs_Air # Abs Coeff = ACS * Density [ 1/cm = cm^2/g * g/cm^3 ]

acs_Be =  openXCOM(XOP_data + '/XOP-2p4_Be.txt',c)            # Absorption Cross Section [cm^2/g]
mu_Be = 1.85 * acs_Be       # Abs Coeff = ACS * Density [ 1/cm = cm^2/g * g/cm^3 ]

acs_KI =  openXCOM(XOP_data + '/XOP-2p4_KI.txt',c)            # Absorption Cross Section [cm^2/g]
mu_KI = 3.12 * acs_KI       # Abs Coeff = ACS * Density [ 1/cm = cm^2/g * g/cm^3 ]

acs_Kap =  openXCOM(XOP_data + '/XOP-2p4_Polyimide.txt',c)    # Absorption Cross Section [cm^2/g]
mu_Kap = 1.42 * acs_Kap     # Abs Coeff = ACS * Density [ 1/cm = cm^2/g * g/cm^3 ]

acs_Water =  openXCOM(XOP_data + '/XOP-2p4_H2O.txt',c)        # Absorption Cross Section [cm^2/g]
mu_Water = 1 * acs_Water    # Abs Coeff = ACS * Density [ 1/cm = cm^2/g * g/cm^3 ]

#%% KI SOLUTION DATA: Cross Sections and Densities
conc = np.array([ 0 , 4 , 20 , 26 , 30 , 50 ])
dens = np.array([ 1 , 1.028 , 1.168 , 1.227 , 1.26 , 1.54 ])

p1 =   9.078e-05    # (5.93e-05, 0.0001223)
p2 =    0.006199    # (0.004601, 0.007796)
p3 =       1.002    # (0.9846, 1.019)

rho_00_KI = 1       # Water
rho_10_KI = p1*10**2 + p2*10 + p3
rho_20_KI = p1*20**2 + p2*20 + p3
rho_30_KI = p1*30**2 + p2*30 + p3
rho_40_KI = p1*40**2 + p2*40 + p3
rho_50_KI = p1*50**2 + p2*50 + p3

mu_00_KI = ( 1.0*acs_Water + 0.0*acs_KI ) * rho_00_KI
mu_10_KI = ( 0.9*acs_Water + 0.1*acs_KI ) * rho_10_KI
mu_20_KI = ( 0.8*acs_Water + 0.2*acs_KI ) * rho_20_KI
mu_30_KI = ( 0.7*acs_Water + 0.3*acs_KI ) * rho_30_KI
mu_40_KI = ( 0.6*acs_Water + 0.4*acs_KI ) * rho_40_KI
mu_50_KI = ( 0.5*acs_Water + 0.5*acs_KI ) * rho_50_KI

#%% PATH LENGTHS [cm] 
L_CsI = 00.0150
L_Be = 00.0000
L_Air = 38.0000
L = np.arange(00.00001,0.501,0.01)     # Variable path length of liquid

#%% SPECTRA & SCINTILLATOR RESPONSE
val = openXCOM('XOP Data/XOP W Tube ' + str(V) + 'keV.txt',1)
x = np.arange(1, len(val)+1,1)
val = interp1d(x, val, kind='linear', fill_value='extrapolate')(keV)    # Interpolate new variable to existing x values so they are consistent
Spec_Pow = val * keV

# SCINTILLATOR ABSORPTION & RESPONSE
Spec_Vis = np.empty(len(keV))
for i in range(len(keV)):
    Spec_Vis[i] = Spec_Pow[i] - (Spec_Pow[i] * np.exp(-mu_CsI[i] * L_CsI)) # Absorbed Spectra by Scintillator

Spec_I_0 = np.empty(len(keV))
# I_0 = Filtered Spectrum
for i in range(len(keV)):
    Spec_I_0[i] = Spec_Vis[i] * np.exp(-mu_Air[i] * L_Air) * np.exp(-mu_Be[i] * L_Be);

#%% SPECTRAL DATA
I_0 = Spec_I_0
I_50_KI = np.empty([len(L),len(keV)])
T_50_KI = np.empty(len(L))
Mu_50_KI = np.empty(len(L))

for j in range(len(L)): # Model
    for i in range(len(keV)):
        I_50_KI[j,i] = I_0[i] * np.exp(-mu_50_KI[i] * L[j])
    
    # Integrating over all energies
    T_50_KI[j] = np.trapz(I_50_KI[j,:],keV) / np.trapz(I_0,keV)
    Mu_50_KI[j] = (-1 / L[j]) * np.log(T_50_KI[j])

order = 9 # polynomial order

x = T_50_KI
y = Mu_50_KI
p = np.polyfit(x,y,order)

# Highest order first
# if order = 3: p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]
np.savetxt(root + '/p50-KI_' + str(V) + 'V_MUvsT.out',p)