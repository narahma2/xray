# -*- coding: utf-8 -*-
"""
Compares the Power & Flux spectra from XOP.
Created on 27 May 2020.

@author: rahmann
"""

import os
import pickle
import numpy as np
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from whitebeam_2018 import (
                            spectra_angles,
                            scint_respfn,
                            filtered_spectra
                            )
from general.Spectra.spectrum_modeling import (
                                               multi_angle as xop,
                                               xcom,
                                               xcom_reshape,
                                               )
from general.misc import create_folder

# Location of APS 2018-1 data
prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'
inp_fld = '{0}/Spectra_Inputs'.format(prj_fld)

def main():
    global prj_fld
    global inp_fld

    # Load XOP spectra
    powr = xop('{0}/xsurface1.dat'.format(inp_fld))
    flux = xop('{0}/ysurface1.dat'.format(inp_fld))

    # Extract energy
    energy = powr['Energy']

    # Find the angles corresponding to the 2018-1 image vertical pixels
    angles_mrad, _ = spectra_angles('{0}/Images/Uniform_Jets/Mean/AVG_'
                                    'Jet_flat2.tif'.format(prj_fld))

    # Create an interpolation object based on angle
    # Passing in an angle in mrad will output an interp spectra (XOP as ref.) 
    powr_linfit = interp1d(
                           x=powr['Angle'],
                           y=powr['Intensity'],
                           axis=0
                           )
    flux_linfit = interp1d(
                           x=flux['Angle'],
                           y=flux['Intensity'],
                           axis=0
                           )

    # Array containing spectra corresponding to the rows of the 2018-1 images
    powr2D = powr_linfit(angles_mrad)
    flux2D = flux_linfit(angles_mrad)

    # Array containing the weighting functions for each row
    wtpowr = np.array([x/np.sum(x) for x in powr2D])
    wtflux = np.array([x/np.sum(x) for x in flux2D])

    # Load NIST XCOM attenuation curves
    YAG_atten = xcom(inp_fld + '/YAG.txt', att_column=3)

    # Reshape XCOM x-axis to match XOP
    YAG_atten = xcom_reshape(YAG_atten, energy)

    # Scintillator EPL
    YAG_epl = 0.05      # 500 um

    # Scintillator densities from Crytur 
    #   <https://www.crytur.cz/materials/yagce/>
    YAG_den =  4.57

    # Apply Beer-Lambert law
    # Scintillator response
    powr_resp = scint_respfn(powr_linfit, YAG_atten, YAG_den, YAG_epl)
    flux_resp = scint_respfn(flux_linfit, YAG_atten, YAG_den, YAG_epl)

    # Apply filters and find detected visible light emission
    YAG = map(lambda x: filtered_spectra(energy, x, powr_resp), powr2D)
    YAG = list(YAG)
    powr_det = np.swapaxes(YAG, 0, 1)[1]

    YAG = map(lambda x: filtered_spectra(energy, x, flux_resp), flux2D)
    YAG = list(YAG)
    flux_det = np.swapaxes(YAG, 0, 1)[1]

    # Plot the weights array
    gs = gridspec.GridSpec(2, 2)

    plt.figure()
    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    plt.plot(
             energy/1000,
             wtpowr[175,:],
             label='Power @ 175',
             color='seagreen',
             linewidth=2.0
             )
    plt.plot(
             energy/1000,
             wtflux[175,:],
             label='Flux @ 175',
             color='cornflowerblue',
             linewidth=2.0,
             linestyle='--',
             zorder=2
             )
    plt.legend()
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Photon Fraction')

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    plt.plot(
             energy/1000,
             wtpowr[60,:],
             label='Power @ 60',
             color='seagreen',
             linewidth=2.0
             )
    plt.plot(
             energy/1000,
             wtflux[60,:],
             label='Flux @ 60',
             color='cornflowerblue',
             linewidth=2.0,
             linestyle='--',
             zorder=2
             )
    plt.legend()
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Photon Fraction')

    ax = plt.subplot(gs[1, :]) # row 1, span all columns
    plt.plot(
             energy/1000,
             powr_det[175,:] / np.max(powr_det[175,:]),
             label='Power @ 60',
             color='seagreen',
             linewidth=2.0
             )
    plt.plot(
             energy/1000,
             flux_det[175,:] / np.max(flux_det[175,:]),
             label='Flux @ 60',
             color='cornflowerblue',
             linewidth=2.0,
             linestyle='--',
             zorder=2
             )
    plt.legend()
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Norm. Intensity')
    plt.savefig('{0}/Figures/xop_compare.png'.format(prj_fld))
    plt.close()


# Run this script
if __name__ == '__main__':
    main()
