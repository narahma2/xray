# -*- coding: utf-8 -*-
"""
Creates the white beam model for the 2018 imaging data sets (KI calibration
jets, solid cone, Aero ECN injector).
Created on Wed Jun 12 14:44:11 2019

@author: rahmann
"""

import os
import pickle
import numpy as np
import warnings

import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from general.nist import mass_atten
from general.spectrum_modeling import (
                                       multi_angle as xop,
                                       xcom_reshape,
                                       density_KIinH2O,
                                       beer_lambert,
                                       visible_light,
                                       beer_lambert_unknown as bl_unk
                                       )
from general.misc import create_folder


# Location of APS 2018-1 data
prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'
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
    cm_pix = np.loadtxt('{0}/cm_px.txt'.format(prj_fld))

    # Distance X-ray beam travels
    length = 3500

    # Convert image vertical locations to angles
    vertical_indices = np.linspace(0, flatfield.shape[0], flatfield.shape[0])
    vertical_indices -= beam_middle_avg
    angles_mrad = np.arctan((vertical_indices*cm_pix) / length) * 1000

    return angles_mrad, flatfield_avg


def plot_atten(atten1, atten2, name):
    """
    Plots the attenuation coefficients before and after re-shaping.
    =============
    --VARIABLES--
    atten1:     Original attenuation coefficient from nist.mass_atten.
    atten2:     Updated attenuation coefficient from
                spectrum_modeling.xcom_reshape.
    name:       Name of filter.
    """
    atten_fld = create_folder('{0}/Figures/Atten_Coeff'.format(prj_fld))

    plt.figure()
    plt.plot(
             atten1['Energy'],
             atten1['Attenuation'],
             label='Original',
             linestyle='solid',
             color='k',
             linewidth=2.0,
             zorder=1
             )
    plt.plot(
             atten2['Energy'],
             atten2['Attenuation'],
             label='Re-shaped',
             linestyle='dashed',
             color='b',
             linewidth=2.0,
             zorder=2
             )
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Attenuation Coefficient (cm$^2$/g')
    plt.title('{0}'.format(name))
    plt.savefig('{0}/{1}.png'.format(atten_fld, name))
    plt.close()

    return


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
             x[0],
             y[0],
             color='k',
             linewidth=2.0,
             label='Water'
             )

    # 1.6% KI
    plt.plot(
             x[1],
             y[1],
             marker='x',
             markevery=50,
             linewidth=2.0,
             label='1.6% KI'
             )

    # 3.4% KI
    plt.plot(
             x[2],
             y[2],
             linestyle='--',
             linewidth=2.0,
             label='3.4% KI'
             )

    # 4.8% KI
    plt.plot(
             x[3],
             y[3],
             linestyle='-.',
             linewidth=2.0,
             label='4.8% KI'
             )

    # 8.0% KI
    plt.plot(
             x[4],
             y[4],
             linestyle=':',
             linewidth=2.0,
             label='8.0% KI'
             )

    # 10.0% KI
    plt.plot(
             x[5],
             y[5],
             marker='^',
             markevery=50,
             linewidth=2.0,
             label='10.0% KI'
             )

    # 11.1% KI
    plt.plot(
             x[6],
             y[6],
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


def filtered_spectra(energy, spectra, scint_resp):
    """
    Applies filters to the X-ray spectra based on 2018-1 setup.
    =============
    --VARIABLES--
    energy:             Energy values from XOP spectra.
    spectra:            Scintillator density (same units as EPL).
    scint_resp:         Scintillator path length (um/mm/cm).
    """
    global inp_fld

    # Load NIST XCOM attenuation curves
    air_atten1 = mass_atten(['Air'], xcom=1, keV=200)
    Be_atten1 = mass_atten(['Be'], xcom=1, keV=200)

    # Reshape XCOM x-axis to match XOP
    air_atten2 = xcom_reshape(air_atten1, energy)
    Be_atten2 = xcom_reshape(Be_atten1, energy)

    # Plot the attenuation coefficients
    plot_atten(air_atten1, air_atten2, 'Air')
    plot_atten(Be_atten1, Be_atten2, 'Be')

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
                                    attenuation=air_atten2['Attenuation'],
                                    density=air_den,
                                    epl=air_epl
                                    )

    # Apply Be window filter
    spectra_filtered = beer_lambert(
                                    incident=spectra_filtered,
                                    attenuation=Be_atten2['Attenuation'],
                                    density=Be_den,
                                    epl=Be_epl
                                    )

    # Apply correction filter (air)
#    spectra_filtered = beer_lambert(
#                                    incident=spectra_filtered,
#                                    attenuation=air_atten2['Attenuation'],
#                                    density=air_den,
#                                    epl=70
#                                    )

    # Find detected spectra
    spectra_det = spectra_filtered * scint_resp

    return spectra_filtered, spectra_det


def spray_model(spray_epl, energy, model, scint, I0, wfct):
    """
    Applies the spray to the filtered X-ray spectra.
    =============
    --VARIABLES--
    spray_epl:  Spray path lengths to iterate through.
    energy:     Energy values from XOP.
    model:      Model type (water/KI%).
    scint:      Scintillator name (LuAG/YAG).
    I0:         Detected flat field spectra.
    wfct:       Weighting function for T calculation.
    """
    global prj_fld
    global inp_fld

    # Spray densities
    if model == 'water':
        ki_perc = 0
        spray_den = 1.0
    elif model == 'KI1p6':
        ki_perc = 1.6
        spray_den = density_KIinH2O(ki_perc)
    elif model == 'KI3p4':
        ki_perc = 3.4
        spray_den = density_KIinH2O(ki_perc)
    elif model == 'KI4p8':
        ki_perc = 4.8
        spray_den = density_KIinH2O(ki_perc)
    elif model == 'KI8p0':
        ki_perc = 8.0
        spray_den = density_KIinH2O(ki_perc)
    elif model == 'KI10p0':
        ki_perc = 10.0
        spray_den = density_KIinH2O(ki_perc)
    elif model == 'KI11p1':
        ki_perc = 11.1
        spray_den = density_KIinH2O(ki_perc)

    # Spray composition
    molec = ['H2O', 'KI']
    comp = [100-ki_perc, ki_perc]

    # Spray attenuation
    liq_atten1 = mass_atten(molec=molec,comp=comp, xcom=1, keV=200)
    liq_atten2 = xcom_reshape(liq_atten1, energy)
    plot_atten(liq_atten1, liq_atten2, model)

    ## Detected spray spectra I 
    I = [
         bl_unk(
                incident=incident,
                attenuation=liq_atten2['Attenuation'],
                density=spray_den,
                epl=spray_epl
                )
         for incident in I0
         ]

    # Swap the axes of I so that it's EPL (len(spray_epl)) x Row (352) x
    #   Intensity (1991)
    I = np.swapaxes(I, 0, 1)

    method = 1
    # Method 1: Ratio of areas
    if method == 1:
        Transmission = np.trapz(y=I, x=energy) / np.trapz(y=I0, x=energy)
    # Method 2: Weighted sum
    elif method == 2:
        Transmission = [np.sum((x/I0)*wfct, axis=1) for x in I]
    # Method 3: Average value
    elif method == 3:
        Transmission = np.mean(I/I0, axis=2)

    # Swap axes so that it's Row x EPL
    Transmission = np.swapaxes(Transmission, 0, 1)

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

    with open('{0}/{1}_model_{2}.pckl'
              .format(mdl_fld, model, scint), 'wb') as f:
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
    # xsurface1: Power, ysurface1: Flux
    inp_spectra = xop('{0}/ysurface1.dat'.format(inp_fld))

    # Extract energy
    energy = inp_spectra['Energy']

    # Find the angles corresponding to the 2018-1 image vertical pixels
    angles_mrad, _ = spectra_angles('{0}/Images/Uniform_Jets/Mean/AVG_'
                                    'Jet_flat2.tif'.format(prj_fld))

    # Create an interpolation object based on angle
    # Passing in an angle in mrad will output an interp spectra (XOP as ref.) 
    sp_linfit = interp1d(
                         x=inp_spectra['Angle'],
                         y=inp_spectra['Intensity'],
                         axis=0
                         )

    # Array containing spectra corresponding to the rows of the 2018-1 images
    spectra2D = sp_linfit(angles_mrad)

    # Array containing the weighting functions for each row
    weights2D = np.array([x/np.sum(x) for x in spectra2D])

    # Load NIST XCOM attenuation curves
    YAG_atten1 = mass_atten(['YAG'], xcom=1, col=3, keV=200)
    LuAG_atten1 = mass_atten(['LuAG'], xcom=1, col=3, keV=200)

    # Reshape XCOM x-axis to match XOP
    YAG_atten2 = xcom_reshape(YAG_atten1, energy)
    LuAG_atten2 = xcom_reshape(LuAG_atten1, energy)

    # Plot attenuation coefficients
    plot_atten(YAG_atten1, YAG_atten2, 'YAG')
    plot_atten(LuAG_atten1, LuAG_atten2, 'LuAG')

    # Scintillator EPL
    YAG_epl = 0.05      # 500 um
    LuAG_epl = 0.01     # 100 um

    # Scintillator densities from Crytur 
    #   <https://www.crytur.cz/materials/yagce/>
    YAG_den =  4.57
    LuAG_den = 6.73

    # Apply Beer-Lambert law
    # Scintillator response
    LuAG_resp = scint_respfn(
                             sp_linfit,
                             scint_atten=LuAG_atten2,
                             scint_den=LuAG_den,
                             scint_epl=LuAG_epl
                             )
    YAG_resp = scint_respfn(
                            sp_linfit,
                            scint_atten=YAG_atten2,
                            scint_den=YAG_den,
                            scint_epl=YAG_epl
                            )

    # Apply filters and find detected visible light emission
    LuAG = map(lambda x: filtered_spectra(energy, x, LuAG_resp), spectra2D)
    LuAG = list(LuAG)
    LuAG_det = np.swapaxes(LuAG, 0, 1)[1]

    YAG = map(lambda x: filtered_spectra(energy, x, YAG_resp), spectra2D)
    YAG = list(YAG)
    YAG_det = np.swapaxes(YAG, 0, 1)[1]

    # Spray EPL
    spray_epl = np.linspace(0.001, 0.82, 820)

    for i, m in enumerate(model):
        [atten_avg_LuAG[i], trans_avg_LuAG[i]] = spray_model(
                                                             spray_epl,
                                                             energy=energy,
                                                             model=m,
                                                             scint='LuAG',
                                                             I0=LuAG_det,
                                                             wfct=weights2D
                                                             )
        [atten_avg_YAG[i], trans_avg_YAG[i]] = spray_model(
                                                           spray_epl,
                                                           energy=energy,
                                                           model=m,
                                                           scint='YAG',
                                                           I0=YAG_det,
                                                           wfct=weights2D
                                                           )

    with open('{0}/Model/avg_var_LuAG.pckl'.format(prj_fld), 'wb') as f:
        pickle.dump([spray_epl, atten_avg_LuAG, trans_avg_LuAG], f)

    with open('{0}/Model/avg_var_YAG.pckl'.format(prj_fld), 'wb') as f:
        pickle.dump([spray_epl, atten_avg_YAG, trans_avg_YAG], f)

    # Plot the averaged attenuation/transmission
    atten_avg = [atten_avg_LuAG, atten_avg_YAG]
    trans_avg = [trans_avg_LuAG, trans_avg_YAG]

    for i, scint in enumerate(['LuAG', 'YAG']):
        averaged_plots(
                       trans_avg[i],
                       atten_avg[i],
                       ylbl='Beam Avg. Atten. Coeff. [1/cm]',
                       xlbl='Transmission',
                       scale='log',
                       name='coeff_vs_trans',
                       scint=scint
                       )
        averaged_plots(
                       np.tile(10*np.array(spray_epl), [7, 1]),
                       atten_avg[i],
                       ylbl='Beam Avg. Atten. Coeff. [1/cm]',
                       xlbl='EPL [mm]',
                       scale='log',
                       name='coeff_vs_epl',
                       scint=scint
                       )
        averaged_plots(
                       np.tile(10*np.array(spray_epl), [7, 1]),
                       1-np.array(trans_avg[i]),
                       ylbl='Attenuation',
                       xlbl='EPL [mm]',
                       scale='linear',
                       name='atten_vs_epl',
                       scint=scint
                       )
        averaged_plots(
                       np.tile(10*np.array(spray_epl), [7, 1]),
                       trans_avg[i],
                       ylbl='Transmission',
                       xlbl='EPL [mm]',
                       scale='linear',
                       name='trans_vs_epl',
                       scint=scint
                       )

    # Plot the weights array
    plt.figure()
    plt.plot(
             energy/1000,
             weights2D[175,:],
             label='y = 175',
             color='black',
             linewidth=2.0
             )
    plt.plot(
             energy/1000,
             weights2D[60,:],
             label='y = 60',
             color='mediumblue',
             linewidth=2.0,
             linestyle='--'
             )
    plt.legend()
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('Photon Fraction')
    plt.savefig('{0}/Figures/weights.png'.format(prj_fld))
    plt.close()

# Run this script
if __name__ == '__main__':
        main()
