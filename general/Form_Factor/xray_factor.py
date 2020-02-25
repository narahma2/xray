# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:03:03 2018

@author: rahmann
"""

import numpy as np
import matplotlib.pyplot as plt
#from nexusformat.nexus import nxload
from scipy import signal

def load_qspace(nexus_file, window=75, plot=False):
    nx = nxload(nexus_file)
    qspace = {"q": nx.entry.result.q.nxvalue, "Intensity": nx.entry.result.data.nxvalue}
    
    intensity_normalized = qspace["Intensity"] / np.sum(qspace["Intensity"])
    intensity_smoothed = signal.savgol_filter(intensity_normalized, window, 3)
    
    atom1_num = (2/3) * waas_kirf("H", qspace["q"])**2
    atom2_num = (1/3) * waas_kirf("O", qspace["q"])**2
    atom1_den = (2/3) * waas_kirf("H", qspace["q"])
    atom2_den = (1/3) * waas_kirf("O", qspace["q"])
    
    structure_factor = ((intensity_smoothed - (atom1_num + atom2_num)) / \
                        (atom1_den + atom2_den)**2)    
    
    if plot:
        plt.figure()
        plt.plot(qspace["q"], structure_factor, label="Structure Factor")    
    
    return {"q": qspace["q"], "I": intensity_normalized, "I Smoothed": intensity_smoothed, \
            "S": structure_factor}


def ItoS(q, I, plot=False):  
    intensity_normalized = I / np.sum(I)
    
    atom1_num = (2/3) * waas_kirf("H", q)**2
    atom2_num = (1/3) * waas_kirf("O", q)**2
    atom1_den = (2/3) * waas_kirf("H", q)
    atom2_den = (1/3) * waas_kirf("O", q)
    
    structure_factor = ((intensity_normalized - (atom1_num + atom2_num)) / \
                        (atom1_den + atom2_den)**2)    
    
    if plot:
        plt.figure()
        plt.plot(q, structure_factor, label="Structure Factor")    
    
    return structure_factor


def waas_kirf(atom, q):
    """Returns the atomic form factor based on the Waasmaier-Kirfel tables using Aikman expansion and modification
    based on charge transfer
    Refs:       a) https://bruceravel.github.io/demeter/pods/Xray/Scattering/WaasKirf.pm.html
                b) doi:10.1038/nature13266, Supplementary Information, Eq. S.23"""
    
    if atom == "H" or atom == "hydrogen":
        file = "E:/General Scripts/python/Form_Factor/hydrogen.txt"
        alpha = -0.43
    elif atom == "O" or atom == "oxygen":
        file = "E:/General Scripts/python/Form_Factor/oxygen.txt"
        alpha = 0.1075
    
    lines = []
    with open(file, 'r') as in_file:
        for i, line in enumerate(in_file):
            if i >= 15:
                lines.append(line.rstrip('\n'))
                    
    coefficients = {k:v for k,v in (x.split(': ') for x in lines)}
    
    f0 = (float(coefficients["a1"]) * np.exp(-float(coefficients["b1"])*q**2) + \
          float(coefficients["a2"]) * np.exp(-float(coefficients["b2"])*q**2) + \
          float(coefficients["a3"]) * np.exp(-float(coefficients["b3"])*q**2) + \
          float(coefficients["a4"]) * np.exp(-float(coefficients["b4"])*q**2) + \
          float(coefficients["a5"]) * np.exp(-float(coefficients["b5"])*q**2)) + float(coefficients["c"])
    
    # Modified atomic form factor
    fp = (1 + alpha * np.exp(-q**2 / (2*2.01))) * f0
    
    return fp