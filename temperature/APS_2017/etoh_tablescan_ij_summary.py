"""
Summarizes the processed IJ Ramping data sets.
See "X-ray Temperature/APS 2017-2/IJ Ethanol Ramping" in OneNote.

Created on Wedn March  18 11:55:00 2020

@author: rahmann
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from general.calc_statistics import polyfit


def main():
    prj_fld = '/mnt/r/X-ray Temperature/APS 2017-2'
    folder = prj_fld + '/Processed/Ethanol'

    # Create summary of Temperature data sets (for most consistent profile)
    flds = glob.glob(folder + '/IJ Ramping/Temperature/*/')
    for fld in flds:
        temp = fld.rsplit('/')[-1]
        files = glob.glob(fld + '/Profiles/profile*')
        names = [
                 x.rsplit('/')[-1].rsplit('_')[-1].rsplit('.')[0]
                 for x in files
                 ]
        df = pd.DataFrame(
                          columns=['R^2', 'Mean', 'StD', 'CfVar', 'CfDisp'],
                          index=names
                          )
        for name, file in zip(names, files):
            # Profile data
            data = np.loadtxt(file)

            # Y positions
            y = np.loadtxt(fld + '/positions.txt')

            # Calculate R^2
            r2 = polyfit(data, y, 1)['determination']

            # Calculate Coefficient of Variation
            mean = np.mean(data)
            std = np.std(data)
            cv = np.std(data) / np.mean(data)

            # Calculate Coefficient of Dispersion
            Q1 = np.percentile(data, 25, interpolation='midpoint')
            Q3 = np.percentile(data, 75, interpolation='midpoint')
            cd = (Q1 - Q3) / (Q1 + Q3)

            # Fill in specific profile in DataFrame
            df.loc[name] = pd.Series({
                                      'R^2': round(r2, 3),
                                      'Mean': round(mean, 3),
                                      'StD': round(std, 3),
                                      'CfVar': round(cv, 3),
                                      'CfDisp': round(cd, 3)
                                      })

        df.to_csv(fld + '/' + temp + 'summary.txt', sep='\t')

    # Create summary of Position data sets (to find best profile)
    flds = glob.glob(folder + '/IJ Ramping/Positions/*/')
    for fld in flds:
        pos = fld.rsplit('/')[-1]
        files = glob.glob(fld + '/Profiles/profile*')
        names = [
                 x.rsplit('/')[-1].rsplit('_')[-1].rsplit('.')[0]
                 for x in files
                 ]
        df = pd.DataFrame(
                          columns=['R^2', 'Mean', 'StD', 'CfVar', 'CfDisp'],
                          index=names
                          )
        for name, file in zip(names, files):
            # Profile data
            data = np.loadtxt(file)

            # Temperature
            T = np.loadtxt(fld + '/temperature.txt')

            # Calculate R^2
            r2 = polyfit(data, T, 1)['determination']

            # Calculate Coefficient of Variation
            mean = np.mean(data)
            std = np.std(data)
            cv = np.std(data) / np.mean(data)

            # Calculate Coefficient of Dispersion
            Q1 = np.percentile(data, 25, interpolation='midpoint')
            Q3 = np.percentile(data, 75, interpolation='midpoint')
            cd = (Q1 - Q3) / (Q1 + Q3)

            # Fill in specific profile in DataFrame
            df.loc[name] = pd.Series({
                                      'R^2': round(r2, 3),
                                      'Mean': round(mean, 3),
                                      'StD': round(std, 3),
                                      'CfVar': round(cv, 3),
                                      'CfDisp': round(cd, 3)
                                      })

        df.to_csv(fld + '/' + pos + 'summary.txt', sep='\t')

    # Summarize the Temperature/Position summaries
    for param in ["Temperature", "Positions"]:
        flds = glob.glob(folder + '/IJ Ramping/' + param + '/?*p*/')

        # Get Temperature/Position values
        y_axis = [
                  float(x.rsplit('/')[-2][1:].replace('p', '.'))
                  for x in flds
                  ]

        # Profile names (kurtosis, A1, q2, etc.)
        profiles = glob.glob(flds[0] + '/Profiles/profile*')
        names = [
                 x.rsplit('/')[-1].rsplit('_')[-1].rsplit('.')[0]
                 for x in profiles
                 ]

        # R^2 summary
        r2_mean = np.array([
                            np.mean([
                                     pd.read_csv(
                                                 fld + '/summary.txt',
                                                 sep='\t',
                                                 index_col=0,
                                                 header=0
                                                 ).loc[name]['R^2']
                                     for fld in flds
                                     ])
                            for name in names
                            ])
        r2_std = np.array([
                           np.std([
                                   pd.read_csv(
                                               fld + '/summary.txt',
                                               sep='\t',
                                               index_col=0,
                                               header=0
                                               ).loc[name]['R^2']
                                   for fld in flds
                                   ])
                           for name in names
                           ])
        r2_cv = r2_std / r2_mean
        r2_q1 = np.array([
                          np.percentile([
                                         pd.read_csv(
                                                     fld + '/summary.txt',
                                                     sep='\t',
                                                     index_col=0,
                                                     header=0
                                                     ).loc[name]['R^2']
                                         for fld in flds
                                         ],
                                        25,
                                        interpolation='midpoint'
                                        )
                          for name in names
                          ])
        r2_q3 = np.array([
                          np.percentile([
                                         pd.read_csv(
                                                     fld + '/summary.txt',
                                                     sep='\t',
                                                     index_col=0,
                                                     header=0
                                                     ).loc[name]['R^2']
                                         for fld in flds
                                         ],
                                        75,
                                        interpolation='midpoint'
                                        )
                          for name in names
                          ])
        r2_cd = (r2_q1 - r2_q3) / (r2_q1 + r2_q3)
        r2_data = {
                   'Mean of R^2': r2_mean,
                   'StD of R^2': r2_std,
                   'CfVar of R^2': r2_cv,
                   'CfDis of R^2': r2_cd
                   }
        df = pd.DataFrame(r2_data, index=names)
        df.to_csv(
                  '{0}/IJ Ramping/{1}_r2_summary.txt'.format(
                                                             folder,
                                                             param.lower()
                                                             ),
                  sep='\t'
                  )

        # Plot R^2 values
        plots_folder = folder + '/IJ Ramping/' + param + '/Plots_R2/'
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        r2_data = [
                   [
                    pd.read_csv(
                                fld + '/summary.txt',
                                sep='\t',
                                index_col=0,
                                header=0
                                ).loc[name]['R^2']
                    for fld in flds
                    ]
                   for name in names
                   ]

        for i, x in enumerate(r2_data):
            plt.figure()
            plt.plot(x, y_axis, ' o')
            plt.xlabel(names[i])
            plt.ylabel(param)
            plt.tight_layout()
            plt.savefig(plots_folder + names[i] + '.png')
            plt.close()

        # Mean summary
        mean_mean = np.array([
                              np.mean([
                                       pd.read_csv(
                                                   fld + '/summary.txt',
                                                   sep='\t',
                                                   index_col=0,
                                                   header=0
                                                   ).loc[name]['Mean']
                                       for fld in flds
                                       ])
                              for name in names
                              ])
        mean_std = np.array([
                             np.std([
                                     pd.read_csv(
                                                 fld + '/summary.txt',
                                                 sep='\t',
                                                 index_col=0,
                                                 header=0
                                                 ).loc[name]['Mean']
                                     for fld in flds
                                     ])
                             for name in names
                             ])
        mean_cv = mean_std / mean_mean
        mean_q1 = np.array([
                            np.percentile([
                                           pd.read_csv(
                                                       fld + '/summary.txt',
                                                       sep='\t',
                                                       index_col=0,
                                                       header=0
                                                       ).loc[name]['Mean']
                                           for fld in flds
                                           ],
                                          25,
                                          interpolation='midpoint'
                                          )
                            for name in names
                            ])
        mean_q3 = np.array([
                            np.percentile([
                                           pd.read_csv(
                                                       fld + '/summary.txt',
                                                       sep='\t',
                                                       index_col=0,
                                                       header=0
                                                       ).loc[name]['Mean']
                                           for fld in flds
                                           ],
                                          75,
                                          interpolation='midpoint'
                                          )
                            for name in names
                            ])
        mean_cd = (mean_q1 - mean_q3) / (mean_q1 + mean_q3)
        mean_data = {
                     'Mean of Mean': mean_mean,
                     'StD of Mean': mean_std,
                     'CfVar of Mean': mean_cv,
                     'CfDis of Mean': mean_cd
                     }
        df = pd.DataFrame(mean_data, index=names)
        df.to_csv(
                  '{0}/IJ Ramping/{1}_mean_summary.txt'.format(
                                                               folder,
                                                               param.lower()
                                                               ),
                  sep='\t'
                  )

        # Plot Mean values
        plots_folder = folder + '/IJ Ramping/' + param + '/Plots_Mean/'
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        mean_data = [
                     [
                      pd.read_csv(
                                  fld + '/summary.txt',
                                  sep='\t',
                                  index_col=0,
                                  header=0
                                  ).loc[name]['Mean']
                      for fld in flds
                      ]
                     for name in names
                     ]

        for i, x in enumerate(mean_data):
            plt.figure()
            plt.plot(x, y_axis, ' o')
            plt.xlabel(names[i])
            plt.ylabel(param)
            plt.tight_layout()
            plt.savefig(plots_folder + names[i] + '.png')
            plt.close()


if __name__ == '__main__':
    main()
