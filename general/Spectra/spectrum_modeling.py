"""
-*- coding: utf-8 -*-

Imports XOP (power) spectrum, which is saved as a .dat file (XYZ).
Returns either  a) multiple spectra for white-beam (vertical variation) setup
                b) singular spectrum

@Author: Naveed (archHarbinger)
@Date:   Tue May 23 16:05:35 2018
@Last Modified by:   Naveed (archHarbinger)
@Last Modified time: 2020-04-03 15:40:47
"""

import pandas as pd
import numpy as np


def multi_angle(xop_path):
    """Used for modeling vertical variation in synchrotron spectra"""
    # [Angle (mrad), Energy (eV), Spectrosc. Power (Watts/eV/mrad)]
    spectrum = pd.read_csv(xop_path, sep='\s+', header=None)
    spectrum.columns = ["Angle", "Energy", "Power"]

    # Can sort the spectrum DataFrame if needed, however will be split into lists instead
    spectrum = spectrum.sort_values(["Angle", "Energy"])
    spectrum = spectrum.reset_index(drop=True)
    
    spectrum["Power"] = spectrum["Power"] * spectrum["Energy"]

    # List of spectrum separated by angle (e.g. if there are 336 unique angles, this returns len(spectrum_list) = 336
    spectrum_list = [v for k, v in spectrum.groupby("Angle")]

    xop_spectrum = {"Angle": [np.mean(item["Angle"]) for item in spectrum_list], "Energy": spectrum_list[0]["Energy"], "Power": [item["Power"] for item in spectrum_list]}

    return xop_spectrum

def tube_xop(xop_path):
    """Used for modeling the tube source spectra"""
    # [Energy (eV), Flux (photons/1keV(bw)/mA/mm^2(@1m)/sec)]
    spectrum = pd.read_csv(xop_path, sep='\s+', names=["Energy", "Flux"], skiprows=4)
    
    # Convert Flux to units of (photons/mA/mm^2(@1m)/sec)
    spectrum["Flux"] = spectrum["Flux"] * (spectrum["Energy"] / 1000)
    
    # Convert Flux to Power in units of (W/mA/mm^2(@1m))
    spectrum["Flux"] = spectrum["Flux"] * (spectrum["Energy"] * (1.602E-19))    # Convert eV to Joules, Watts = J/s
    spectrum.columns = ["Energy", "Power"]
    
    return spectrum

def xcom(xcom_path, att_column=3):
    xcom_spectra = pd.read_csv(xcom_path, sep='\t')

    # Convert abscissa from MeV to eV
    xcom_spectra.iloc[:, 0] = xcom_spectra.iloc[:, 0] * 1E6

    # Add residual value to edge locations for interpolation
    for i in range(len(xcom_spectra.iloc[:, 0]) - 1):
        if xcom_spectra.iloc[i, 0] == xcom_spectra.iloc[i + 1, 0]:
            xcom_spectra.iloc[i, 0] -= 1E-6
            xcom_spectra.iloc[i + 1, 0] += 1E-6

    # Rename column name
    xcom_spectra = xcom_spectra.rename(columns={'Photon Energy (MeV)': 'Photon Energy (eV)'})

    # Energy is in eV, Attenuation is in cm^2/g
    xcom_spectra = {"Energy": xcom_spectra.iloc[:, 0], "Attenuation": xcom_spectra.iloc[:, att_column]}

    return xcom_spectra


def xcom_reshape(xcom_spectra, xop_abscissa):
    """Linear interpolation to reconfigure the x-axis of the xcom spectra to match XOP input"""
    y = np.interp(xop_abscissa, xcom_spectra["Energy"], xcom_spectra["Attenuation"])
    y = {"Energy": xop_abscissa, "Attenuation": y}

    return y


def beer_lambert(incident, attenuation, density, epl):
    """Beer-Lambert Law with attenuation coefficient on a mass basis, outputs EPL in [length]"""
    transmitted = incident * np.exp(-attenuation * density * epl)

    return transmitted


def visible_light(incident, scint_trans):
    """Converts scintillator transmission to scintillator absorption (image on camera)"""
    vl = [y1 - y2 for (y1,y2) in zip(incident, scint_trans)]
    
    return vl


def beer_lambert_unknown(incident, attenuation, density, epl):
    transmitted = [incident * np.exp(-attenuation * density * length) for length in epl]
    
    return transmitted


def density_KIinH2O(concentration):
    """Returns the density of known KI concentration in g/cm^3"""
    conc = [0, 4, 20, 26, 30, 50]
    dens = [1, 1.028, 1.168, 1.227, 1.26, 1.54]
    
    p1 = 0.078E-5
    p2 = 0.006199
    p3 = 1.002
    T = 50
    f = p1*T**2 + p2*T + p3
    
    # Density of 10% KI in water by mass
    # rho_10_KI = p1*10^2 + p2*10 + p3
    
    rho = p1*(concentration)**2 + p2*(concentration) + p3
    
    return rho
