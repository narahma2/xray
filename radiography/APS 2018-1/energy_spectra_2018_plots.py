"""
-*- coding: utf-8 -*-
Creates energy spectra plots for the whitebeam 2018-1 model.

@Author: rahmann
@Date:   2020-04-20 12:27:25
@Last Modified by:   rahmann
@Last Modified time: 2020-04-20 12:27:25
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from general.nist import mass_atten
from general.spectrum_modeling import multi_angle as xop, xcom, xcom_reshape
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from whitebeam_2018 import spectra_angles, scint_respfn, filtered_spectra
from general.misc import create_folder

def main():
    # Location of APS 2018-1 data
    prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'

    # Load XOP spectra
    input_folder = prj_fld + '/Spectra_Inputs'
    input_spectra = xop(input_folder + '/xsurface1.dat')

    # Get energy values
    energy = input_spectra['Energy']

    # Find the angles corresponding to the 2018-1 image vertical pixels
    angles_mrad, flatfield_avg = spectra_angles('{0}/Images/Uniform_Jets/Mean/AVG_'
                                    'Jet_flat2.tif'.format(prj_fld))

    # Find the index that best approximates middle (angle = 0)
    middle_index = np.argmin(abs(angles_mrad))

    # Create an interpolation object based on angle
    # Passing in an angle in mrad will output an interpolated spectra (w/ XOP as reference) 
    spectra_linfit = interp1d(input_spectra['Angle'], input_spectra['Intensity'], axis=0)

    # Create an array containing spectra corresponding to each row of the 2018-1 images
    spectra2D = spectra_linfit(angles_mrad)

    # Middle spectra
    spectra_middle = spectra2D[middle_index, :]

    # Load NIST XCOM attenuation curves
    YAG_atten = mass_atten(['YAG'], xcom=1, col=3, keV=200)
    LuAG_atten = mass_atten(['LuAG'], xcom=1, col=3, keV=200)

    # Reshape XCOM x-axis to match XOP
    YAG_atten = xcom_reshape(YAG_atten, energy)
    LuAG_atten = xcom_reshape(LuAG_atten, energy)

    # Scintillator EPL
    YAG_epl = 0.05      # 500 um
    LuAG_epl = 0.01     # 100 um

    ## Density in g/cm^3
    # Scintillator densities from Crytur <https://www.crytur.cz/materials/yagce/>
    YAG_den =  4.57
    LuAG_den = 6.73

    ## Apply Beer-Lambert Law
    # Find detected scintillator spectrum (along middle) and response curves (no filters)
    LuAG_resp = scint_respfn(spectra_linfit, LuAG_atten, LuAG_den, LuAG_epl)
    YAG_resp = scint_respfn(spectra_linfit, YAG_atten, YAG_den, YAG_epl)

    # Find detected spectra (no filters)
    LuAG_detected_nofilters = np.array([x*LuAG_resp for x in spectra2D])
    YAG_detected_nofilters = np.array([x*YAG_resp for x in spectra2D])

    # Use filters
    LuAG = map(lambda x: filtered_spectra(energy, x, LuAG_resp), spectra2D)
    LuAG = list(LuAG)
    spectra_filtered = np.swapaxes(LuAG, 0, 1)[0]
    LuAG_detected = np.swapaxes(LuAG, 0, 1)[1]

    YAG = map(lambda x: filtered_spectra(energy, x, YAG_resp), spectra2D)
    YAG = list(YAG)
    YAG_detected = np.swapaxes(YAG, 0, 1)[1]

    ## Plot figures
    plot_folder = create_folder(prj_fld + '/Figures/Energy_Spectra')

    # Create vertical integrated power curves
    integrated_xray = np.trapz(spectra2D, energy, axis=1)
    integrated_xray_filtered = np.trapz(spectra_filtered, energy, axis=1)
    integrated_LuAG = np.trapz(LuAG_detected_nofilters, energy, axis=1)
    integrated_YAG = np.trapz(YAG_detected_nofilters, energy, axis=1)
    integrated_LuAG_filtered = np.trapz(LuAG_detected, energy)
    integrated_YAG_filtered = np.trapz(YAG_detected, energy)

    # Integrated plots
    plt.figure()
    plt.plot(integrated_xray, color='black', linewidth=1.5, label='X-ray Beam')
    plt.plot(integrated_xray_filtered, color='black', linestyle='--', linewidth=1.5, label='X-ray Beam (filtered)')
    plt.plot(integrated_LuAG, color='cornflowerblue', linewidth=1.5, label='LuAG')
    plt.plot(integrated_LuAG_filtered, color='cornflowerblue', linestyle='--', linewidth=1.5, label='LuAG (filtered)')
    plt.plot(integrated_YAG, color='seagreen', linewidth=1.5, label='YAG')
    plt.plot(integrated_YAG_filtered, color='seagreen', linestyle='--', linewidth=1.5, label='YAG (filtered)')
    plt.legend()
    plt.savefig(plot_folder + '/integrated_intensity.png')
    plt.close()

    # Integrated and normalized (filtered) plots
    plt.figure()
    plt.plot(integrated_xray_filtered / max(integrated_xray_filtered), zorder=1, color='black', linewidth=1.5, label='X-ray Beam (filtered)')
    plt.plot(integrated_LuAG_filtered / max(integrated_LuAG_filtered), zorder=2, color='cornflowerblue', linewidth=1.5, label='LuAG (filtered)')
    plt.plot(integrated_YAG_filtered / max(integrated_YAG_filtered), zorder=3, color='seagreen', linewidth=1.5, label='YAG (filtered)')
    plt.plot(flatfield_avg / max(flatfield_avg), zorder=4, color='red', linewidth=1.5, label='Flat Field')
    plt.legend()
    plt.savefig(plot_folder + '/integrated_norm_intensity.png')
    plt.close()

    # Scintillator response
    plt.figure()
    plt.plot(energy / 1000, LuAG_resp, linewidth=1.5, color='cornflowerblue', label='LuAG')
    plt.plot(energy / 1000, YAG_resp, linewidth=1.5, color='seagreen', label='YAG')
    plt.legend()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Scintillator Response')
    plt.savefig(plot_folder + '/scintillator_response.png')
    plt.close()

    # Spectra along middle
    plt.figure()
    plt.plot(energy / 1000, spectra_middle, linewidth=1.5, color='black', label='Incident')
    plt.plot(energy / 1000, spectra_filtered[middle_index, :], linewidth=1.5, color='black', linestyle='--', label='Incident (filtered)')
    plt.plot(energy / 1000, LuAG_detected_nofilters[middle_index, :], linewidth=1.5, color='cornflowerblue', label='LuAG')
    plt.plot(energy / 1000, LuAG_detected[middle_index, :], linewidth=1.5, color='cornflowerblue', linestyle='--', label='LuAG (filtered)')
    plt.plot(energy / 1000, YAG_detected_nofilters[middle_index, :], linewidth=1.5, color='seagreen', label='YAG')
    plt.plot(energy / 1000, YAG_detected[middle_index, :], linewidth=1.5, color='seagreen', linestyle='--', label='YAG (filtered)')
    plt.legend()
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intensity')
    plt.savefig(plot_folder + '/middle_spectra.png')
    plt.close()

    # LuAG filtered spectra at various locations
    plt.figure()
    plt.plot(energy / 1000, LuAG_detected[middle_index, :], linewidth=1.5, color='black', label='{0}'.format(middle_index))
    plt.plot(energy / 1000, LuAG_detected[middle_index+50, :], linewidth=1.5, color='blue', label='{0}'.format(middle_index+50))
    plt.plot(energy / 1000, LuAG_detected[middle_index-50, :], linewidth=1.5, color='blue', linestyle='--', label='{0}'.format(middle_index-50))
    plt.plot(energy / 1000, LuAG_detected[middle_index+100, :], linewidth=1.5, color='red', label='{0}'.format(middle_index+100))
    plt.plot(energy / 1000, LuAG_detected[middle_index-100, :], linewidth=1.5, color='red', linestyle='--', label='{0}'.format(middle_index-100))
    plt.legend()
    plt.title('Filtered')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intensity')
    plt.savefig(plot_folder + '/LuAG_filtered_spectra.png')
    plt.close()

    # YAG filtered spectra at various locations
    plt.figure()
    plt.plot(energy / 1000, YAG_detected[middle_index, :], linewidth=1.5, color='black', label='{0}'.format(middle_index))
    plt.plot(energy / 1000, YAG_detected[middle_index+50, :], linewidth=1.5, color='blue', label='{0}'.format(middle_index+50))
    plt.plot(energy / 1000, YAG_detected[middle_index-50, :], linewidth=1.5, color='blue', linestyle='--', label='{0}'.format(middle_index-50))
    plt.plot(energy / 1000, YAG_detected[middle_index+100, :], linewidth=1.5, color='red', label='{0}'.format(middle_index+100))
    plt.plot(energy / 1000, YAG_detected[middle_index-100, :], linewidth=1.5, color='red', linestyle='--', label='{0}'.format(middle_index-100))
    plt.legend()
    plt.title('Filtered')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intensity')
    plt.savefig(plot_folder + '/YAG_filtered_spectra.png')
    plt.close()

    # X-ray spectra at various locations
    plt.figure()
    plt.plot(energy / 1000, spectra2D[middle_index, :], linewidth=1.5, color='black', label='{0}'.format(middle_index))
    plt.plot(energy / 1000, spectra2D[middle_index+50, :], linewidth=1.5, color='blue', label='{0}'.format(middle_index+50))
    plt.plot(energy / 1000, spectra2D[middle_index-50, :], linewidth=1.5, color='blue', linestyle='--', label='{0}'.format(middle_index-50))
    plt.plot(energy / 1000, spectra2D[middle_index+100, :], linewidth=1.5, color='red', label='{0}'.format(middle_index+100))
    plt.plot(energy / 1000, spectra2D[middle_index-100, :], linewidth=1.5, color='red', linestyle='--', label='{0}'.format(middle_index-100))
    plt.legend()
    plt.title('Unfiltered')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intensity')
    plt.savefig(plot_folder + '/xray_spectra.png')
    plt.close()


# Process this script
if __name__ == '__main__':
    main()
