"""
Compares the interpolated IJ Ramping temperatures from the calibration
to the Nozzle T. Constructs a thermometer using the IJ Ramping/Positions
data sets.


@Author: narahma2
@Date:   2020-03-23 13:06:21
@Last Modified by:   narahma2
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from general.calc_statistics import polyfit


def main():
    # Setup initial parameters
    prj_fld = '/mnt/r/X-ray Temperature/APS 2017-2'
    fld = prj_fld + '/Processed/Ethanol'

    # Select profiles to be used as thermometers
    profiles = ['aratio', 'peak', 'peakq', 'var', 'skew', 'kurt', 'pca']

    # Select calibration data sets
    calibrations = ['408', '409', 'Combined']

    # Load in IJ Ramping folders
    flds = glob.glob(fld + '/IJ Ramping/Positions/y*/')

    # Load in and process data sets
    # Iterate through each selected calibration jet
    for calibration in calibrations:
        # Initialize summary arrays
        summary_rmse = np.nan * np.zeros((len(flds), len(profiles)))
        summary_mape = np.nan * np.zeros((len(flds), len(profiles)))
        summary_zeta = np.nan * np.zeros((len(flds), len(profiles)))
        summary_mdlq = np.nan * np.zeros((len(flds), len(profiles)))

        # Iterate through each selected profile
        for j, profile in enumerate(profiles):
            # Create polynomial object based on selected profile & calib jet
            # Load APS 2018-1 for 'Combined'
            if calibration == 'Combined':
                cmb_fld = fld.replace('APS 2017-2', 'APS 2018-1')\
                             .replace('Ethanol', 'Ethanol_700umNozzle')
                p = np.poly1d(np.loadtxt(cmb_fld + '/' + calibration +
                              '/Statistics/' + profile + '_polynomial.txt'))
            else:
                p = np.poly1d(np.loadtxt(fld + '/' + calibration +
                              '/Statistics/' + profile + '_polynomial.txt'))

            # Create flds
            plots_fld = '{0}/IJ Ramping/PositionsInterp/{1}_{2}'\
                        .format(fld, calibration, profile)
            if not os.path.exists(plots_fld):
                os.makedirs(plots_fld)

            summary_nozzleT = []
            summary_interpT = []

            # Load in Positions
            pos_y = []
            for i, fldr in enumerate(flds):
                # Y position string (y00p25, y00p05, etc.)
                yp = fldr.rsplit('/')[-2]
                pos_y.append(float(yp[1:].replace('p', '.')))

                # Profile data for the IJ Ramping Positions
                data = np.loadtxt('{0}/Profiles/profile_{1}.txt'
                                  .format(fldr, profile)
                                  )

                # Nozzle T
                nozzleT = np.loadtxt(fldr + '/temperature.txt')
                # Interpolated temperature
                interpT = p(data)
                # Fit a linear line of interpT vs. nozzleT
                fit = polyfit(interpT, nozzleT, 1)

                # Calculate RMSE (root mean squared error)
                rmse = np.sqrt(((interpT - nozzleT)**2).mean())

                # Calculate MAPE (mean absolute percentage error)
                mape = 100*(np.sum(np.abs((interpT - nozzleT) / nozzleT)) /
                            len(nozzleT))

                # Calculate median symmetric accuracy
                zeta = 100*np.exp(np.median(np.abs(np.log(interpT/nozzleT)))
                                  - 1)

                # Calculate MdLQ (median log accuracy ratio)
                # Positive/negative: systematic (over/under)-prediction
                mdlq = np.median(np.log(interpT/nozzleT))

                # Build up summaries
                summary_rmse[i, j] = rmse
                summary_mape[i, j] = mape
                summary_zeta[i, j] = zeta
                summary_mdlq[i, j] = mdlq
                summary_nozzleT.append(nozzleT)
                summary_interpT.append(interpT)

                # Plot results
                plt.figure()
                plt.plot(
                         interpT,
                         nozzleT,
                         ' o',
                         markerfacecolor='none',
                         markeredgecolor='b',
                         label='Data'
                         )
                plt.plot(
                         nozzleT,
                         nozzleT,
                         'k',
                         linewidth=2.0,
                         label='y = x'
                         )
                plt.plot(
                         interpT,
                         fit['function'](interpT),
                         'r',
                         linewidth=2.0,
                         label='y = ' + '%0.2f'%fit['polynomial'][0] + 'x + '
                               + '%0.2f'%fit['polynomial'][1]
                         )
                plt.title('y = ' + yp[1:].replace('p', '.') + ' mm - ' +
                          calibration + ': ' + profile)
                plt.legend()
                plt.xlabel('Interpolated Temperature (K)')
                plt.ylabel('Nozzle Temperature (K)')
                plt.tight_layout()
                plt.savefig(plots_fld + '/' + yp + '.png')
                plt.close()

            slices = [1, 3, 5, 7, 9, 11, 12, 13]
            plt.figure()
            [
             plt.plot(
                      summary_nozzleT[i],
                      summary_interpT[i],
                      linewidth=2.0,
                      label=str(pos_y[i]) + ' mm'
                      )
             for i in slices
             ]
            plt.legend()
            plt.xlim([270, 350])
            plt.ylim([270, 350])
            plt.ylabel('Interpolated Temperature (K)')
            plt.xlabel('Nozzle Temperature (K)')
            plt.title(calibration + ': ' + profile)
            plt.savefig(plots_fld + '/temperatures.png')
            plt.close()

        # Plot summaries
        plt.figure()
        [
         plt.plot(
                  pos_y,
                  summary_rmse[:, j],
                  linewidth=2.0,
                  label=profiles[j]
                  )
         for j in range(len(profiles))
         ]
        plt.legend()
        plt.ylabel('RMSE (K)')
        plt.xlabel('Vertical Location (mm)')
        plt.title(calibration)
        plt.savefig(fld + '/IJ Ramping/PositionsInterp/' + calibration +
                    '_rmse.png')
        plt.close()

        plt.figure()
        [
         plt.plot(
                  pos_y,
                  summary_mape[:, j],
                  linewidth=2.0,
                  label=profiles[j]
                  )
         for j in range(len(profiles))
         ]
        plt.legend()
        plt.ylabel('MAPE (%)')
        plt.xlabel('Vertical Location (mm)')
        plt.title(calibration)
        plt.savefig(fld + '/IJ Ramping/PositionsInterp/' + calibration +
                    '_mape.png')
        plt.close()

        plt.figure()
        [
         plt.plot(
                  pos_y,
                  summary_zeta[:, j],
                  linewidth=2.0,
                  label=profiles[j]
                  )
         for j in range(len(profiles))
         ]
        plt.legend()
        plt.ylabel(r'$\zeta$ (%)')
        plt.xlabel('Vertical Location (mm)')
        plt.title(calibration)
        plt.savefig(fld + '/IJ Ramping/PositionsInterp/' + calibration +
                    '_zeta.png')
        plt.close()

        plt.figure()
        [
         plt.plot(
                  pos_y,
                  summary_mdlq[:, j],
                  linewidth=2.0,
                  label=profiles[j]
                  )
         for j in range(len(profiles))
         ]
        plt.legend()
        plt.ylabel('MdLQ (-)')
        plt.xlabel('Vertical Location (mm)')
        plt.title(calibration)
        plt.savefig(fld + '/IJ Ramping/PositionsInterp/' + calibration +
                    '_mdlq.png')
        plt.close()

        # Save summary file
        np.savetxt(fld + '/IJ Ramping/PositionsInterp/' + calibration +
                   '_rmse.txt', summary_rmse, delimiter='\t',
                   header='\t'.join(str(x) for x in profiles))
        np.savetxt(fld + '/IJ Ramping/PositionsInterp/' + calibration +
                   '_mape.txt', summary_mape, delimiter='\t',
                   header='\t'.join(str(x) for x in profiles))
        np.savetxt(fld + '/IJ Ramping/PositionsInterp/' + calibration +
                   '_zeta.txt', summary_zeta, delimiter='\t',
                   header='\t'.join(str(x) for x in profiles))
        np.savetxt(fld + '/IJ Ramping/PositionsInterp/' + calibration +
                   '_mdlq.txt', summary_mdlq, delimiter='\t',
                   header='\t'.join(str(x) for x in profiles))


if __name__ == '__main__':
    main()
