# @Author: rahmann
# @Date:   2020-04-10 16:38:31
# @Last Modified by:   rahmann
# @Last Modified time: 2020-04-28 22:21:29

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
    calibrations = ['408', '409']

    # Load in IJ Ramping folders
    flds = ['/IJ Cold']

    # Load in and process data sets
    # Iterate through each selected profile
    for calibration in calibrations:
        for j, profile in enumerate(profiles):
            # Create polynomial object based on selected profile & calib jet
            # Load APS 2018-1 for 'Combined'
            p = np.poly1d(
                          np.loadtxt(
                                     '{0}/{1}/Statistics/{2}_polynomial.txt'
                                     .format(folder, calibration, profile)
                                     )
                          )

            # Load in Positions
            for i, fld in enumerate(flds):
                # Create folders
                plots_folder = folder + fld + ' Interp/' + calibration
                if not os.path.exists(plots_folder):
                    os.makedirs(plots_folder)

                # Positions array
                positions = np.loadtxt('{0}{1}/Profiles/positions.txt'
                                       .format(folder, fld)
                                       )

                # Profile data for the IJ Ramping Positions
                data = np.loadtxt('{0}{1}/Profiles/profile_{2}.txt'
                                  .format(folder, fld, profile)
                                  )

                # Nozzle T (3 deg C)
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
                plt.title('T = 276 K - ' + calibration + ': ' + profile)
                plt.legend()
                plt.xlabel('Y Location (mm)')
                plt.ylabel('Interpolated Temperature (K)')
                plt.tight_layout()
                plt.ylim([270, 300])
                plt.savefig(plots_folder + '/' + profile + '.png')
                plt.close()


if __name__ == '__main__':
    main()
