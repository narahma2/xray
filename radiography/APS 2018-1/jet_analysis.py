"""
-*- coding: utf-8 -*-
Summarize the transmission correction factors for the jets.

@Author: rahmann
@Date:   2020-04-30 10:54:07
@Last Modified by:   rahmann
@Last Modified time: 2020-04-30 10:54:07
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from general.calc_statistics import polyfit
from general.misc import create_folder


def get_xpos(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    xpos = np.nanmean(processed_data['Lateral Position'])

    return xpos


def get_mean_elpsT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    mean_elpsT = np.nanmean(processed_data['Transmission Ratios'][0])

    return mean_elpsT


def get_elpsT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    elpsT = np.array(processed_data['Transmission Ratios'][0])

    return elpsT


def get_mean_peakT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    mean_peakT = np.nanmean(processed_data['Transmission Ratios'][1])

    return mean_peakT


def get_peakT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    peakT = np.array(processed_data['Transmission Ratios'][1])

    return peakT


def main():
    # Location of APS 2018-1 data
    prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'

    # Save location for the plots
    plots_folder = create_folder('{0}/Figures/Jet_Summary/'.format(prj_fld))

    # Scintillator
    scintillators = ['LuAG', 'YAG']

    # KI %
    KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]

    # Test matrix
    test_matrix = pd.read_csv(
                              '{0}/APS White Beam.txt'.format(prj_fld),
                              sep='\t+',
                              engine='python'
                              )

    # Crop down the test matrix
    test_matrix = test_matrix[['Test', 'Nozzle Diameter (um)', 'KI %']].copy()

    for scint in scintillators:
        # Processed data sets location
        prc_fld = '{0}/Processed/{1}/Summary/'.format(prj_fld, scint)

        # Vertical variation
        vert_mat = test_matrix[~test_matrix['Test'].str.contains('mm')].copy()

        # Groups
        grp1 = 'Nozzle Diameter (um)'
        grp2 = 'KI %'
        rpkT = 'Ratio Peak T'
        relT = 'Ratio Ellipse T'

        # Get vertical values
        vert_mat[relT] = [
                          get_elpsT('{0}/{1}_{2}.pckl'
                                    .format(prc_fld, scint, x))
                          for x in vert_mat['Test']
                          ]
        vert_mat[rpkT] = [
                          get_peakT('{0}/{1}_{2}.pckl'
                                    .format(prc_fld, scint, x))
                          for x in vert_mat['Test']
                          ]
        vert_peakT_grp = vert_mat.groupby([grp1, grp2])\
                                 .apply(lambda x: np.mean(x[rpkT], axis=0))
        vert_elpsT_grp = vert_mat.groupby([grp1, grp2])\
                                 .apply(lambda x: np.mean(x[relT], axis=0))
        axial_loc = np.linspace(start=20, stop=325, num=325-20+1, dtype=int)

        linecolors = ['dimgray', 'firebrick', 'goldenrod', 'mediumseagreen',
                      'steelblue', 'mediumpurple', 'hotpink']
        linelabels = ['0%', '1.6%', '3.4%', '4.8%', '8%', '10%', '11.1%']

        warnings.filterwarnings('ignore')
        # Vertical peakT plot
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.set_size_inches(12, 6)
        for i, x in enumerate(linecolors):
            ax1.plot(
                     axial_loc[2:-1],
                     vert_peakT_grp[700, KI_conc[i]][2:-1],
                     color=x,
                     linewidth=2.0
                     )
            ax2.plot(
                     axial_loc[2:-1],
                     vert_peakT_grp[2000, KI_conc[i]][2:-1],
                     color=x,
                     linewidth=2.0
                     )
        ax1.title.set_text(r'700 $\mu$m')
        ax2.title.set_text(r'2000 $\mu$m')
        ax1.set_xlabel('Axial Position (px)')
        ax2.set_xlabel('Axial Position (px)')
        ax1.set_ylabel('Correction Factor (-)')
        fig.suptitle('Vertical Variation - Ratio Peak T')
        fig.legend(
                   ax1,
                   labels=linelabels,
                   loc='center right',
                   borderaxespad=0.1,
                   title=grp2
                   )
        plt.subplots_adjust(wspace=0.05, top=0.90)
        plt.savefig('{0}/{1}_vert_peakT.png'.format(plots_folder, scint))
        plt.close()

        # Vertical elpsT plot
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.set_size_inches(12, 6)
        for i, x in enumerate(linecolors):
            ax1.plot(
                     axial_loc[2:-1],
                     vert_elpsT_grp[700, KI_conc[i]][2:-1],
                     linewidth=2.0
                     )
            ax2.plot(
                     axial_loc[2:-1],
                     vert_elpsT_grp[2000, KI_conc[i]][2:-1],
                     linewidth=2.0
                     )
        ax1.title.set_text(r'700 $\mu$m')
        ax2.title.set_text(r'2000 $\mu$m')
        ax1.set_xlabel('Axial Position (px)')
        ax2.set_xlabel('Axial Position (px)')
        ax1.set_ylabel('Correction Factor (-)')
        fig.suptitle('Vertical Variation - Ratio Ellipse T')
        fig.legend(
                   ax1,
                   labels=linelabels,
                   loc='center right',
                   borderaxespad=0.1,
                   title=grp2
                   )
        plt.subplots_adjust(wspace=0.05, top=0.90)
        plt.savefig('{0}/{1}_vert_elpsT.png'.format(plots_folder, scint))
        warnings.filterwarnings('default')

        ######################################################################

        # Horizontal variation
        horiz_matrix = test_matrix[
                                   test_matrix['Test'].str.contains('mm')
                                   ].copy()

        horiz_matrix['X Position'] = [
                                      get_xpos('{0}/{1}_{2}.pckl'
                                               .format(prc_fld, scint, x))
                                      for x in horiz_matrix['Test']
                                      ]

        # Sort horiz_matrix by X position, re-index, and drop the outliers
        horiz_matrix.sort_values(by=['X Position'], inplace=True)
        horiz_matrix.reset_index(inplace=True)
        horiz_matrix.drop([0, len(horiz_matrix)-1], inplace=True)

        # Get horizontal values
        horiz_matrix[relT] = [
                              get_mean_elpsT('{0}/{1}_{2}.pckl'
                                             .format(prc_fld, scint, x))
                              for x in horiz_matrix['Test']
                              ]
        horiz_matrix[rpkT] = [
                              get_mean_peakT('{0}/{1}_{2}.pckl'
                                             .format(prc_fld, scint, x))
                              for x in horiz_matrix['Test']
                              ]

        # Normalize the horiz values to remove KI% dependency
        horiz_matrix[relT] /= max(horiz_matrix[relT])
        horiz_matrix[rpkT] /= max(horiz_matrix[rpkT])

        # Fit a quadratic to the horizontal values
        relT_quad = np.poly1d(
                              np.polyfit(
                                         x=horiz_matrix['X Position'],
                                         y=horiz_matrix[relT],
                                         deg=2
                                         )
                              )
        rpkT_quad = np.poly1d(
                              np.polyfit(
                                         x=horiz_matrix['X Position'],
                                         y=horiz_matrix[rpkT],
                                         deg=2
                                         )
                              )

        # Horizontal plot
        plt.figure()
        plt.plot(
                 horiz_matrix['X Position'],
                 horiz_matrix[rpkT],
                 color='olivedrab',
                 marker='s',
                 label=rpkT
                 )
        plt.plot(
                 horiz_matrix['X Position'],
                 rpkT_quad(horiz_matrix['X Position']),
                 color='black',
                 marker='s',
                 linestyle='dashed',
                 alpha=0.5,
                 label='Peak Fit'
                 )
        plt.legend()
        plt.title('{0} - Horizontal Variation - 700 um, 10% KI'.format(scint))
        plt.savefig('{0}/{1}_horiz_peak.png'.format(plots_folder, scint))
        plt.close()

        plt.figure()
        plt.plot(
                 horiz_matrix['X Position'],
                 horiz_matrix[relT],
                 fillstyle='none',
                 color='olivedrab',
                 marker='s',
                 label=relT
                 )
        plt.plot(
                 horiz_matrix['X Position'],
                 relT_quad(horiz_matrix['X Position']),
                 fillstyle='none',
                 color='black',
                 marker='s',
                 linestyle='dashed',
                 alpha=0.5,
                 label='Ellipse Fit'
                 )
        plt.legend()
        plt.title('{0} - Horizontal Variation - 700 um, 10% KI'.format(scint))
        plt.savefig('{0}/{1}_horiz_elps.png'.format(plots_folder, scint))
        plt.close()

        ######################################################################

        # Mean vertical variation
        mean_matrix = test_matrix[
                                  ~test_matrix['Test'].str.contains('mm')
                                  ].copy()

        # Get mean vertical values
        mean_matrix[relT] = [
                             get_mean_elpsT('{0}/{1}_{2}.pckl'
                                            .format(prc_fld, scint, x))
                             for x in mean_matrix['Test']
                             ]
        mean_matrix[rpkT] = [
                             get_mean_peakT('{0}/{1}_{2}.pckl'
                                            .format(prc_fld, scint, x))
                             for x in mean_matrix['Test']
                             ]

        # Calculate the mean values as needed
        pivot_mean_peakT = mean_matrix.pivot_table(
                                                   values=rpkT,
                                                   index=[grp2],
                                                   columns=[grp1],
                                                   aggfunc=np.nanmean
                                                   )
        pivot_mean_elpsT = mean_matrix.pivot_table(
                                                   values=relT,
                                                   index=[grp2],
                                                   columns=[grp1],
                                                   aggfunc=np.nanmean
                                                   )
        # Create arrays from the pivot tables
        # Could plot directly from the pivot table but I didn't want to delve
        # too deep into that
        mean_peakT_700 = pivot_mean_peakT[700]
        peakT_700_fit = polyfit(KI_conc, mean_peakT_700, 1)
        peakT_700_fit_r2 = peakT_700_fit['determination']
        peakT_700_fit_lbl = 'y$_{700}$ = ' + '{0:0.3f}x + {1:0.3f};'\
                            'R$^2$ {2:0.0f}%'\
                            .format(
                                    peakT_700_fit['polynomial'][0],
                                    peakT_700_fit['polynomial'][1],
                                    100*peakT_700_fit_r2
                                    )

        mean_peakT_2000 = pivot_mean_peakT[2000]
        peakT_2000_fit = polyfit(KI_conc, mean_peakT_2000, 1)
        peakT_2000_fit_r2 = peakT_2000_fit['determination']
        peakT_2000_fit_lbl = 'y$_{2000}$ = ' + '{0:0.3f}x + {1:0.3f};'\
                             'R$^2$ {2:0.0f}%'\
                             .format(
                                     peakT_2000_fit['polynomial'][0],
                                     peakT_2000_fit['polynomial'][1],
                                     100*peakT_2000_fit_r2
                                     )

        mean_elpsT_700 = pivot_mean_elpsT[700]
        elpsT_700_fit = polyfit(KI_conc, mean_elpsT_700, 1)
        elpsT_700_fit_r2 = elpsT_700_fit['determination']
        elpsT_700_fit_lbl = 'y$_{700}$ = ' + '{0:0.3f}x + {1:0.3f};'\
                            'R$^2$ {2:0.0f}%'\
                            .format(
                                    elpsT_700_fit['polynomial'][0],
                                    elpsT_700_fit['polynomial'][1],
                                    100*elpsT_700_fit_r2
                                    )

        mean_elpsT_2000 = pivot_mean_elpsT[2000]
        elpsT_2000_fit = polyfit(KI_conc, mean_elpsT_2000, 1)
        elpsT_2000_fit_r2 = elpsT_2000_fit['determination']
        elpsT_2000_fit_lbl = 'y$_{2000}$ = ' + '{0:0.3f}x + {1:0.3f};'\
                             'R$^2$ {2:0.0f}%'\
                             .format(
                                     elpsT_2000_fit['polynomial'][0],
                                     elpsT_2000_fit['polynomial'][1],
                                     100*elpsT_2000_fit_r2
                                     )

        # PeakT plot (markers filled)
        plt.figure()
        plt.plot(
                 KI_conc,
                 mean_peakT_700,
                 color='olivedrab',
                 marker='s',
                 label='700 um'
                 )
        plt.plot(
                 KI_conc,
                 peakT_700_fit['function'](KI_conc),
                 color='teal',
                 label=peakT_700_fit_lbl
                 )
        plt.plot(
                 KI_conc,
                 mean_peakT_2000,
                 color='indianred',
                 marker='^',
                 label='2000 um'
                 )
        plt.plot(
                 KI_conc,
                 peakT_2000_fit['function'](KI_conc),
                 color='darkorange',
                 label=peakT_2000_fit_lbl
                 )
        plt.legend()
        plt.title('{0} - Ratio Peak T'.format(scint))
        plt.xlabel('KI (%)')
        plt.ylabel('Correction Factor (-)')
        plt.savefig('{0}/{1}_mean_peakT.png'.format(plots_folder, scint))
        plt.close()

        # EllipseT plot (markers not filled)
        plt.figure()
        plt.plot(
                 KI_conc,
                 mean_elpsT_700,
                 fillstyle='none',
                 color='olivedrab',
                 marker='s',
                 label='700 um'
                 )
        plt.plot(
                 KI_conc,
                 elpsT_700_fit['function'](KI_conc),
                 color='teal',
                 label=elpsT_700_fit_lbl
                 )
        plt.plot(
                 KI_conc,
                 mean_elpsT_2000,
                 fillstyle='none',
                 color='indianred',
                 marker='^',
                 label='2000 um'
                 )
        plt.plot(
                 KI_conc,
                 elpsT_2000_fit['function'](KI_conc),
                 color='darkorange',
                 label=elpsT_2000_fit_lbl
                 )
        plt.legend()
        plt.title('{0} - Ratio Ellipse T'.format(scint))
        plt.xlabel('KI (%)')
        plt.ylabel('Correction Factor (-)')
        plt.savefig('{0}/{1}_mean_elpsT.png'.format(plots_folder, scint))
        plt.close()

        ######################################################################

        mean_peakT_combi = np.mean([mean_peakT_700, mean_peakT_2000], axis=0)
        peakT_combi_fit = polyfit(KI_conc, mean_peakT_combi, 1)
        peakT_combi_fit_r2 = peakT_combi_fit['determination']
        peakT_combi_fit_lbl = 'y = {0:0.3f}x + {1:0.3f};'\
                              'R$^2$ {2:0.0f}%'\
                              .format(peakT_combi_fit['polynomial'][0],
                                      peakT_combi_fit['polynomial'][1],
                                      100*peakT_combi_fit_r2
                                      )

        mean_elpsT_combi = np.mean(
                                      [mean_elpsT_700, mean_elpsT_2000],
                                      axis=0
                                      )
        elpsT_combi_fit = polyfit(KI_conc, mean_elpsT_combi, 1)
        elpsT_combi_fit_r2 = elpsT_combi_fit['determination']
        elpsT_combi_fit_lbl = 'y = {0:0.3f}x + {1:0.3f}; R$^2$ {2:0.0f}%'\
                              .format(
                                      elpsT_combi_fit['polynomial'][0],
                                      elpsT_combi_fit['polynomial'][1],
                                      100*elpsT_combi_fit_r2
                                      )

        # Save the linear fitted correction factors
        with open('{0}/Processed/{1}/{1}_peakT_cf.txt'
                  .format(prj_fld, scint), 'wb') as f:
            np.savetxt(f, peakT_combi_fit['function'](KI_conc))

        with open('{0}/Processed/{1}/{1}_elpsT_cf.txt'
                  .format(prj_fld, scint), 'wb') as f:
            np.savetxt(f, elpsT_combi_fit['function'](KI_conc))

        # Map out the correction factor horizontally over an image array
        cf_fld = create_folder('{0}/Processed/{1}/CF_Map/'
                               .format(prj_fld, scint))
        image_x = np.linspace(0, 767, 768)
        for KI in KI_conc:
            KIstr = str(KI).replace('.', 'p')

            # Create CF based on elpsT
            elps_mat = np.ones((352, 768)) * elpsT_combi_fit['function'](KI)
            elps_mat *= relT_quad(image_x)
            elps_im = Image.fromarray(elps_mat)
            elps_im.save('{0}/elps_{1}.tif'.format(cf_fld, KIstr))

            # Create CF based on peakT
            peak_mat = np.ones((352, 768)) * peakT_combi_fit['function'](KI)
            peak_mat *= rpkT_quad(image_x)
            peak_im = Image.fromarray(peak_mat)
            peak_im.save('{0}/peak_{1}.tif'.format(cf_fld, KIstr))

        plt.figure()
        plt.plot(
                 KI_conc,
                 mean_peakT_combi,
                 color='lightcoral',
                 marker='s',
                 linestyle='',
                 label=rpkT,
                 zorder=2
                 )
        plt.plot(
                 KI_conc,
                 peakT_combi_fit['function'](KI_conc),
                 linestyle='-',
                 color='maroon',
                 label=peakT_combi_fit_lbl,
                 zorder=1
                 )
        plt.plot(
                 KI_conc,
                 mean_elpsT_combi,
                 color='cornflowerblue',
                 marker='^',
                 linestyle='',
                 label=relT,
                 zorder=2
                 )
        plt.plot(
                 KI_conc,
                 elpsT_combi_fit['function'](KI_conc),
                 linestyle='-',
                 color='mediumblue',
                 label=elpsT_combi_fit_lbl,
                 zorder=1
                 )
        plt.title('{0} - Combined'.format(scint))
        plt.legend()
        plt.xlabel('KI (%)')
        plt.ylabel('Correction Factor (-)')
        plt.savefig('{0}/{1}_combi.png'.format(plots_folder, scint))
        plt.close()


if __name__ == '__main__':
    main()
