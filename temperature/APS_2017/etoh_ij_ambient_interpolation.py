"""
-*- coding: utf-8 -*-
Interpolation for the IJ Ambient data set.
@Author: naveed
@Date:   2020-04-07 16:49:03
@Last Modified by:   naveed
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Setup initial parameters
    prj_fld = '/mnt/r/X-ray Temperature/APS 2017-2'
    folder = prj_fld + '/Processed/Ethanol'

    # Select profiles to be used as thermometers
    profiles = ['aratio', 'peak', 'peakq', 'var', 'skew', 'kurt', 'pca']

    # Select calibration data sets
    calibration = 'RampUp'

    # Load in IJ Ramping folders
    flds = ['/IJ Ambient']

    # Load in and process data sets
    # Iterate through each selected profile
    for j, profile in enumerate(profiles):
        # Create polynomial object based on selected profile & calibration jet
        # Load APS 2018-1 for 'Combined'
        p = np.poly1d(
                      np.loadtxt('{0}/{1}/Statistics/{2}_polynomial.txt'
                                 .format(folder.replace(
                                                        'APS 2017-2',
                                                        'APS 2018-1'
                                                        )
                                               .replace(
                                                        'Ethanol',
                                                        'Ethanol_700umNozzle'
                                                        ),
                                         calibration,
                                         profile
                                         )
                                 )
                      )

        # Load in Positions
        for i, fld in enumerate(flds):
            # Create folders
            plots_folder = folder + fld + ' Interp'
            if not os.path.exists(plots_folder):
                os.makedirs(plots_folder)

            # Positions array
            positions = np.loadtxt(folder + fld + '/positions.txt')

            # Profile data for the IJ Ramping Positions
            data = np.loadtxt('{0}{1}/Profiles/profile_{2}.txt'
                              .format(folder, fld, profile)
                              )

            # Nozzle T (65 deg C)
            # Interpolated temperature
            interpT = p(data)

            # Plot results
            plt.figure()
            plt.plot(
                     positions,
                     interpT,
                     '-o',
                     markerfacecolor='none',
                     markeredgecolor='b',
                     label='Data'
                     )
            plt.title('T = 298 K - ' + calibration + ': ' + profile)
            plt.legend()
            plt.xlabel('Y Location (mm)')
            plt.ylabel('Interpolated Temperature (K)')
            plt.tight_layout()
            plt.ylim([280, 350])
            plt.savefig(plots_folder + '/' + profile + '.png')
            plt.close()


if __name__ == '__main__':
    main()
