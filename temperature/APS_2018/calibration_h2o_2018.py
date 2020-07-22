"""
Processes the APS 2018-1 temperature data sets.

Created on Thu Jan 16 17:08:33 2020

@author: rahmann
"""

import glob
import numpy as np
import os
from scipy.constants import convert_temperature
from scipy.signal import savgol_filter
from general.xray_factor import ItoS
from temperature_processing import main as temperature_processing

def main():
    # Setup
    prj_fld = '/mnt/r/X-ray Temperature/APS 2018-1/'

    test = 'Water_700umNozzle'

    folder = prj_fld + 'Processed/' + test
    if not os.path.exists(folder):
        os.makedirs(folder)

    scans = ['/RampUp', '/RampDown', '/Combined']

    for scan in scans:
        # Load background
        bg = glob.glob(prj_fld + '/Q Space/Water_700umNozzle/*bkgd*')
        q = np.loadtxt(bg[0], usecols=0)
        bg = [np.loadtxt(x, usecols=1) for x in bg]

        # Average background intensities
        bg = np.mean(bg, axis=0)

        # Intensity correction
        rampUp = [1079, 1080, 1081, 1082, 1083, 1084, 1085, 1087, 1088]
        rampDown = [1089, 1090, 1092, 1093, 1095, 1096, 1097,
                    1098, 1099, 1100]
        combined = rampUp + rampDown

        if 'Up' in scan:
            files = rampUp
        elif 'Down' in scan:
            files = rampDown
        else:
            files = combined

        files = [
                 glob.glob(
                           '{0}/Q Space/Water_700umNozzle/*{1}*'
                           .format(prj_fld, x)
                           )
                 for x in files
                 ]

        # Set q range
        sl = slice(
                   (np.abs(np.array(q) - 1.70)).argmin(),
                   (np.abs(np.array(q) - 3.1)).argmin()
                   )
        reduced_q = q[sl]

        # Load in temperatures and sort
        temperature = np.array([float(x[0].rsplit('Target')[-1].rsplit('_')[0].replace('p','.')) for x in files])

        # Convert temperature to Kelvin
        temperature = convert_temperature(temperature, 'Celsius', 'Kelvin')

        # Load in and process intensity curves (sort, filter, crop, normalize, ItoS)
        intensity = np.array([np.mean([np.loadtxt(x, usecols=1)-bg for x in y], axis=0) for y in files])
        filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]
        reduced_intensity = [x[sl] for x in filtered_intensity]
        #reduced_intensity = np.array([y/np.trapz(y, x=reduced_q) for y in reduced_intensity])
        structure_factor = np.array([ItoS(np.array(reduced_q), x) for x in reduced_intensity])

        temperature_processing(test, folder, scan, reduced_intensity, reduced_q, temperature, structure_factor)


if __name__ == '__main__':
    main()
