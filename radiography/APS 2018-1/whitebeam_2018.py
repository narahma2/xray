# -*- coding: utf-8 -*-
"""
Creates the white beam model for the 2018 imaging data sets (KI calibration
jets, solid cone, Aero ECN injector).
Created on Wed Jun 12 14:44:11 2019

@author: rahmann
"""

import sys
import os
import pickle
import numpy as np
import warnings

import _paths
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from Spectra.spectrum_modeling import (
    multi_angle as xop,
    xcom,
    xcom_reshape,
    density_KIinH2O,
    beer_lambert,
    visible_light,
    beer_lambert_unknown as bl_unk
    )
from jet_processing import create_folder


# Location of APS 2018-1 data
prj_fld = '{0}/X-ray Radiography/APS 2018-1/'.format(sys_folder)
inp_fld = '{0}/Spectra_Inputs'.format(prj_fld)

def spectra_angles(flat):
    """
    Finds the center of the X-ray beam.
    =============
    --VARIABLES--
    flat:          Path to the flat field .TIFF file.
    """
    # Load in the flat field into an array
    flatfield = np.array(Image.open(flat))

    # Average the flat field horizontally
    flatfield_avg = np.mean(flatfield, axis=1)

    # Initialize the array to find the vertical middle of the beam
    beam_middle = np.zeros((flatfield.shape[1]))

    # Loop each column of the image to find the vertical middle
    for i in range(flatfield.shape[1]):
        warnings.filterwarnings('ignore')
        beam_middle[i] = np.argmax(savgol_filter(flatfield[:,i], 55, 3))
        warnings.filterwarnings('default')

    # Take the mean to be the center
    beam_middle_avg = int(np.mean(beam_middle).round())

    # Image pixel size
    cm_pix = 0.16 / 162

    # Distance X-ray beam travels
    length = 3500

    # Convert image vertical locations to angles
    vertical_indices = np.linspace(0, flatfield.shape[0], flatfield.shape[0])
    vertical_indices -= beam_middle_avg
    angles_mrad = np.arctan((vertical_indices*cm_pix) / length) * 1000

    return angles_mrad, flatfield_avg


def averaged_plots(x, y, ylbl, xlbl, scale, name, scint):
    """
    Creates plots of the averaged variables.
    =============
    --VARIABLES--
    x:              X axis variable.
    y:              Y axis variable.
    ylbl:           Y axis label.
    xlbl:           X axis label.
    scale:          Y scale ('log' or 'linear').
    name:           Save name for the plot.
    scint:          Scintillator being used.
    """
    global prj_fld

    averaged_folder = create_folder(
                                    '{0}/Figures/Averaged_Figures'
                                    .format(prj_fld)
                                    )

    plt.figure()

    # Water
    plt.plot(
             x= [0],
             y=y[0],
             color='k',
             linewidth=2.0,
             label='Water'
             )

    # 1.6% KI
    plt.plot(
             x=x[1],
             y=y[1],
             marker='x',
             markevery=50,
             linewidth=2.0,
             label='1.6% KI'
             )

    # 3.4% KI
    plt.plot(
             x=x[2],
             y=y[2],
             linestyle='--',
             linewidth=2.0,
             label='3.4% KI'
             )

    # 4.8% KI
    plt.plot(
             x=x[3],
             y=y[3],
             linestyle='-.',
             linewidth=2.0,
             label='4.8% KI'
             )

    # 8.0% KI
    plt.plot(
             x=x[4],
             y=y[4],
             linestyle=':',
             linewidth=2.0,
             label='8.0% KI'
             )

    # 10.0% KI
    plt.plot(
             x=x[5],
             y=y[5],
             marker='^',
             markevery=50,
             linewidth=2.0,
             label='10.0% KI'
             )

    # 11.1% KI
    plt.plot(
             x=x[6],
             y=y[6],
             linestyle='--',
             linewidth=2.0,
             label='11.1% KI'
             )
    plt.legend()
    plt.ylabel(ylbl)
    plt.xlabel(xlbl)
    plt.yscale(scale)
    plt.savefig('{0}/{1}_{2}.png'.format(averaged_folder, scint, name))


def scint_respfn(sp_linfit, scint_atten, scint_den, scint_epl):
    """
    Constructs the scintillator response function. Response curves were
    verified in a previous commit to be the same vertically (center = edge).
    =============
    --VARIABLES--
    sp_linfit:          Linfit object that returns spectra based on Y.
    scint_atten:        XCOM scintillator attenuation coefficient curve.
    scint_den:          Scintillator density (same units as EPL).
    scint_epl:          Scintillator path length (um/mm/cm).
    """
    # Use middle spectra for finding response
    spectra_middle = sp_linfit(0)

    # Find the transmitted spectrum through the scintillator
    scint_trans = beer_lambert(
                               incident=spectra_middle,
                               attenuation=scint_atten['Attenuation'],
                               density=scint_den,
                               epl=scint_epl
                               )

    # Find the detected visible light emission from the scintillator
    det = visible_light(spectra_middle, scint_trans)

    # Find the scintillator response
    scint_resp = det / spectra_middle

    return scint_resp


def filtered_spectra(spectra, scint_resp):
    """
    Applies filters to the X-ray spectra based on 2018-1 setup.
    =============
    --VARIABLES--
    spectra:            Scintillator density (same units as EPL).
    scint_resp:         Scintillator path length (um/mm/cm).
    """
    global inp_fld

    # Load NIST XCOM attenuation curves
    air_atten = xcom(inp_fld + '/air.txt', att_column=5)
    Be_atten = xcom(inp_fld + '/Be.txt', att_column=5)

    # Reshape XCOM x-axis to match XOP
    air_atten = xcom_reshape(air_atten, inp_spectra['Energy'])
    Be_atten = xcom_reshape(Be_atten, inp_spectra['Energy'])

    ## EPL in cm
    # Air EPL
    air_epl = 70

    # Be window EPL
    # See Alan's 'Spray Diagnostics at the Advanced Photon Source 
    #   7-BM Beamline' (3 Be windows)
    Be_epl = 0.075

    ## Density in g/cm^3
    # Air density
    air_den = 0.001275

    # Beryllium density
    Be_den = 1.85

    # Apply air filter
    spectra_filtered = beer_lambert(
                                    incident=spectra,
                                    attenuation=air_atten['Attenuation'],
                                    density=air_den,
                                    epl=air_epl
                                    )

    # Apply Be window filter
    spectra_filtered = beer_lambert(
                                    incident=spectra_filtered,
                                    attenuation=Be_atten['Attenuation'],
                                    density=Be_den,
                                    epl=Be_epl
                                    )

    # Apply correction filter (air)
#    spectra_filtered = beer_lambert(
#                                    incident=spectra_filtered,
#                                    attenuation=air_atten['Attenuation'],
#                                    density=air_den,
#                                    epl=70
#                                    )

    # Find detected spectra
    spectra_det = spectra_filtered * scint_resp

    return spectra_filtered, spectra_det


def spray_model(model, scint, det, I0):
    """
    Applies the spray to the filtered X-ray spectra.
    =============
    --VARIABLES--
    model:      Model type (water/KI%).
    scint:      Scintillator name (LuAG/YAG).
    det:        Detected spectra from the scintillator.
    I0:         Flat field total intensity.
    """
    global prj_fld
    global inp_fld

    # Spray attenuation curves
    liq_atten = xcom('{0}/{1}.txt'.format(inp_fld, m), att_column=5)
    liq_atten = xcom_reshape(liquid_atten, inp_spectra['Energy'])

    # Spray densities
    if model == 'water':
        spray_den = 1.0                         # KI 0%
    elif model == 'KI1p6':
        spray_den = density_KIinH2O(1.6)        # KI 1.6%
    elif model == 'KI3p4':
        spray_den = density_KIinH2O(3.4)        # KI 3.4%
    elif model == 'KI4p8':
        spray_den = density_KIinH2O(4.8)        # KI 4.8%
    elif model == 'KI8p0':
        spray_den = density_KIinH2O(8)          # KI 8.0%
    elif model == 'KI10p0':
        spray_den = density_KIinH2O(10)         # KI 10.0%
    elif model == 'KI11p1':
        spray_den = density_KIinH2O(11.1)       # KI 11.1%

    # Spray EPL
    spray_epl = np.linspace(0.001, 0.82, 820)

    ## Add in the spray
    spray_det = [
                 bl_unk(
                        incident=incident,
                        attenuation=liq_atten['Attenuation'],
                        density=spray_den, spray_epl
                        )
                 for incident in det
                 ]

    # Spray
    I = [np.trapz(x, inp_spectra['Energy']) for x in spray_det]

    ## LHS of Beer-Lambert Law
    Transmission = [x1/x2 for (x1, x2) in zip(I, I0)]

    # Cubic spline fitting of Transmission and spray_epl curves 
    #   Needs to be reversed b/c of monotonically increasing
    #   restriction on 'x', however this doesn't change the interp call.
    # Function that takes in I/I0 and outputs expected EPL (cm)
    TtoEPL = [
              CubicSpline(vertical_pix[::-1], spray_epl[::-1])
              for vertical_pix in Transmission
              ]

    # Function that takes in EPL (cm) value and outputs expected I/I0
    EPLtoT = [
              CubicSpline(spray_epl, vertical_pix)
              for vertical_pix in Transmission
              ]

    # Save model
    mdl_fld = create_folder('{0}/Model/'.format(prj_fld))

    with open('{0}/{1}_model_{2}.pckl'.format(mdl_fld, m, scint), 'wb') as f:
        pickle.dump([TtoEPL, EPLtoT, spray_epl, Transmission], f)

    # Calculate average attenuation and transmission
    atten_avg = np.nanmean([
                            -np.log(x)/spray_epl
                            for x in Transmission
                            ],
                           axis=0
                           )
    trans_avg = np.nanmean(Transmission, axis=0)

    return atten_avg, trans_avg


def main():
    global inp_fld

    model = ['water', 'KI1p6', 'KI3p4', 'KI4p8', 'KI8p0', 'KI10p0', 'KI11p1']
    atten_avg_LuAG = len(model) * [None]
    trans_avg_LuAG = len(model) * [None]
    atten_avg_YAG = len(model) * [None]
    trans_avg_YAG = len(model) * [None]

    # Load XOP spectra
    inp_spectra = xop('{0}/xsurface1.dat'.format(inp_fld))

    # Find the angles corresponding to the 2018-1 image vertical pixels
    angles_mrad, _ = spectra_angles()

    # Create an interpolation object based on angle
    # Passing in an angle in mrad will output an interp spectra (XOP as ref.) 
    sp_linfit = interp1d(
                         x=inp_spectra['Angle'],
                         y=inp_spectra['Power'],
                         axis=0
                         )

    # Array containing spectra corresponding the to rows of the 2018-1 images
    spectra2D = sp_linfit(angles_mrad)

    # Load NIST XCOM attenuation curves
    YAG_atten = xcom(inp_fld + '/YAG.txt', att_column=3)
    LuAG_atten = xcom(inp_fld + '/Al5Lu3O12.txt', att_column=3)

    # Reshape XCOM x-axis to match XOP
    YAG_atten = xcom_reshape(YAG_atten, inp_spectra['Energy'])
    LuAG_atten = xcom_reshape(LuAG_atten, inp_spectra['Energy'])

    # Scintillator EPL
    YAG_epl = 0.05      # 500 um
    LuAG_epl = 0.01     # 100 um

    # Scintillator densities from Crytur 
    #   <https://www.crytur.cz/materials/yagce/>
    YAG_den =  4.57
    LuAG_den = 6.73

    # Apply Beer-Lambert law
    # Scintillator response
    LuAG_response = scint_respfn(
                                 sp_linfit,
                                 attenuation=LuAG_atten,
                                 density=LuAG_den,
                                 epl=LuAG_epl
                                 )
    YAG_response = scint_respfn(sp_linfit, YAG_atten, YAG_den, YAG_epl)

    # Apply filters and find detected visible light emission
    LuAG = map(lambda x: filtered_spectra(x, LuAG_response), spectra2D)
    LuAG = list(LuAG)
    LuAG_det = np.swapaxes(LuAG, 0, 1)[1]

    YAG = map(lambda x: filtered_spectra(x, YAG_response), spectra2D)
    YAG = list(YAG)
    YAG_det = np.swapaxes(YAG, 0, 1)[1]

    ## Total intensity calculations
    # Flat field
    I0_LuAG = [np.trapz(x, inp_spectra['Energy']) for x in LuAG_det]
    I0_YAG = [np.trapz(x, inp_spectra['Energy']) for x in YAG_det]

    for i, m in enumerate(model):
        [atten_avg_LuAG[i], trans_avg_LuAG[i]] = spray_model(
                                                             model=m,
                                                             scint='LuAG',
                                                             det=LuAG_det,
                                                             I0=I0_LuAG
                                                             )
        [atten_avg_YAG[i], trans_avg_YAG[i]] = spray_model(
                                                           model=m,
                                                           scint='YAG',
                                                           det=YAG_det,
                                                           I0=I0_YAG
                                                           )

    with open('{0}/Model/avg_var_LuAG.pckl'.format(prj_fld), 'wb') as f:
        pickle.dump([spray_epl, atten_avg_LuAG, trans_avg_LuAG], f)

    with open('{0}/Model/avg_var_YAG.pckl'.format(prj_fld), 'wb') as f:
        pickle.dump([spray_epl, atten_avg_YAG, trans_avg_YAG], f)

    # Plot
    atten_avg = [atten_avg_LuAG, atten_avg_YAG]
    trans_avg = [trans_avg_LuAG, trans_avg_YAG]

    for i, scint in enumerate(['LuAG', 'YAG']):
        averaged_plots(
                       x=trans_avg[i],
                       y=atten_avg[i],
                       ylbl='Beam Avg. Atten. Coeff. [1/cm]',
                       xlbl='Transmission',
                       scale='log',
                       name='coeff_vs_trans',
                       scint=scint
                       )
        averaged_plots(
                       x=np.tile(10*np.array(spray_epl), [7, 1]),
                       y=atten_avg[i],
                       ylbl='Beam Avg. Atten. Coeff. [1/cm]',
                       xlbl='EPL [mm]',
                       scale='log',
                       name='coeff_vs_epl',
                       scint=scint
                       )
        averaged_plots(
                       x=np.tile(10*np.array(spray_epl), [7, 1]),
                       y=1-np.array(trans_avg[i]),
                       ylbl='Attenuation',
                       xlbl='EPL [mm]',
                       scale='linear',
                       name='atten_vs_epl',
                       scint=scint
                       )
        averaged_plots(
                       x=np.tile(10*np.array(spray_epl), [7, 1]),
                       y=trans_avg[i],
                       ylbl='Transmission',
                       xlbl='EPL [mm]',
                       scale='linear',
                       name='trans_vs_epl',
                       scint=scint
                       )


# Run this script
if __name__ == '__main__':
        main()
