"""
Processes the APS 2018-1 ethanol jet data sets.
Created on Thu Jan 23 17:42:32 2020

@author: rahmann
"""

import glob
import numpy as np
from scipy.constants import convert_temperature
from scipy.signal import savgol_filter
from general.misc import create_folder
from temperature.temperature_processing import main as temperature_processing


def main():
    # Setup
    prj_fld = '/mnt/r/X-ray Temperature/APS 2018-1/'

    test = 'Ethanol_700umNozzle'

    fld = create_folder('{0}/Processed/{1}'.format(prj_fld, test))

    scans = ['/RampUp', '/RampDown', '/Combined']

    for scan in scans:
        # Load background
        bg = glob.glob(prj_fld + '/Q Space/Ethanol_700umNozzle/*Scan1112*')
        # Load q
        q = np.loadtxt(bg[0], usecols=0)
        # Load bg intensity
        bg = [np.loadtxt(x, usecols=1) for x in bg]

        # Average background intensities
        bg = np.mean(bg, axis=0)

        # Intensity correction
        rampUp = [1109, 1111, 1113, 1115, 1117, 1118, 1119, 1120]
        rampDown = [1121, 1122, 1124, 1125, 1126, 1128, 1129, 1130]
        combined = rampUp + rampDown

        if 'Up' in scan:
            files = rampUp
        elif 'Down' in scan:
            files = rampDown
        else:
            files = combined

        files = [
                 glob.glob('{0}/Q Space/Ethanol_700umNozzle/*{1}*'
                           .format(prj_fld, x))
                 for x in files
                 ]

        # Set q range
        sl = slice(
                   (np.abs(np.array(q) - 0.6)).argmin(),
                   (np.abs(np.array(q) - 1.75)).argmin()
                   )
        reduced_q = q[sl]

        # Load in temperatures and sort
        T = np.array([
                      float(x[0].rsplit('Target')[-1].rsplit('_')[0])
                      for x in files
                      ])

        # Convert temperature to Kelvin
        T = convert_temperature(T, 'Celsius', 'Kelvin')

        # Load in and process intensity curves (sort, filter, crop, normalize)
        intensity = np.array([
                              np.mean([
                                       np.loadtxt(x, usecols=1)-bg
                                       for x in y
                                       ],
                                      axis=0)
                              for y in files
                              ])
        filtered_I = [savgol_filter(x, 49, 3) for x in intensity]
        reduced_I = [x[sl] for x in filtered_I]
        #reduced_I = np.array([
        #                      y/np.trapz(y, x=reduced_q)
        #                      for y in reduced_I
        #                      ])

        # Run processing script
        temperature_processing(test, fld, scan, reduced_I, reduced_q, T)


if __name__ == '__main__':
    main()
