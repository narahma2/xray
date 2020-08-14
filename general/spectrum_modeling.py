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

import numpy as np
import pandas as pd


def multi_angle(xop_path):
    """Used for modeling vertical variation in synchrotron spectra"""
    # [Angle (mrad), Energy (eV), Power (Watts/eV/mrad) or Flux
    #   (Photons/s/0.1%bw/mrad)]
    spct = pd.read_csv(xop_path, sep='\s+', header=None, engine='python')
    spct.columns = ['Angle', 'Energy', 'Intensity']

    # Can sort DataFrame if needed, however will be split into lists instead
    spct = spct.sort_values(['Angle', 'Energy'])
    spct = spct.reset_index(drop=True)

    # Convert energy axis to keV
    spct.loc[:, 'Energy'] /= 1000

    # Scale intensity to visible photon count
    # Logic being that 10 keV x-ray photons emit 10x more visible photons 
    #   than 1 keV x-ray photons
    spct['Intensity'] = spct['Intensity'] * spct['Energy']

    # List of spectrum separated by angle 
    # e.g. if there are 336 unique angles, this has len(spct_list) = 336
    spct_list = [v for k, v in spct.groupby('Angle')]

    xop_spct = {
                'Angle': [np.mean(item['Angle']) for item in spct_list],
                'Energy': spct_list[0]['Energy'],
                'Intensity': [item['Intensity'] for item in spct_list]
                }

    return xop_spct


def tube_xop(xop_path):
    """Used for modeling the tube source spectra from XOP"""
    # [Energy (eV), Flux (photons/1keV(bw)/mA/mm^2(@1m)/sec)]
    spectrum = pd.read_csv(
                           xop_path,
                           sep='\s+',
                           names=['Energy', 'Intensity'],
                           skiprows=4
                           )

    # Convert energy to keV
    spectrum['Energy'] /= 1000

    # Convert Flux to units of (photons/mA/mm^2(@1m)/sec)
    spectrum['Intensity'] *= spectrum['Energy']

    return spectrum


def tube_spekcalc(spekcalc_path):
    """Used for modeling the tube source spectra from SpekCalc"""
    # Energy: keV, Intensity: Photons/keV/cm^2/mAs
    spectrum = pd.read_csv(
                           spekcalc_path,
                           skiprows=18,
                           sep='  ',
                           engine='python',
                           names=['Energy', 'Intensity']
                           )

    # Scale intensity to visible photon count
    # Logic being that 10 keV x-ray photons emit 10x more visible photons 
    #   than 1 keV x-ray photons
    spectrum['Intensity'] *= spectrum['Energy']

    return spectrum


def xcom(xcom_path, att_column=3):
    xcom_spct = pd.read_csv(xcom_path, sep='\t')

    # Convert abscissa from MeV to eV
    xcom_spct.iloc[:, 0] = xcom_spct.iloc[:, 0] * 1E6

    # Add residual value to edge locations for interpolation
    for i in range(len(xcom_spct.iloc[:, 0]) - 1):
        if xcom_spct.iloc[i, 0] == xcom_spct.iloc[i + 1, 0]:
            xcom_spct.iloc[i, 0] -= 1E-6
            xcom_spct.iloc[i + 1, 0] += 1E-6

    # Rename column name
    xcom_spct = xcom_spct.rename(columns={'Photon Energy (MeV)':
                                          'Photon Energy (eV)'})

    # Energy is in eV, Attenuation is in cm^2/g
    xcom_spct = {
                 'Energy': xcom_spct.iloc[:, 0],
                 'Attenuation': xcom_spct.iloc[:, att_column]
                 }

    return xcom_spct


def xcom_reshape(xcom_spct, source_energy):
    """Linear interp. to reconfigure the x-axis of the xcom spectra to XOP"""
    y = np.interp(
                  source_energy,
                  xcom_spct['Energy'],
                  xcom_spct['Attenuation']
                  )

    y = {'Energy': source_energy, 'Attenuation': y}

    return y


def beer_lambert(incident, attenuation, density, epl):
    """Beer-Lambert with atten. coeff. on mass basis, EPL in [length]"""
    T = incident * np.exp(-attenuation * density * epl)

    return T


def visible_light(incident, scint_trans):
    """Converts scint. transmission to scint. absorption image on camera)"""
    vl = [y1 - y2 for (y1,y2) in zip(incident, scint_trans)]

    return vl


def beer_lambert_unknown(incident, attenuation, density, epl):
    T = [incident * np.exp(-attenuation * density * length) for length in epl]

    return T


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
