# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:04:28 2019

@author: rahmann
"""


import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import (
                          savgol_filter,
                          find_peaks,
                          peak_widths
                          )
from skimage.transform import rotate
from general.Statistics.calc_statistics import (
                                                rmse,
                                                mape,
                                                zeta,
                                                mdlq
                                                )
from general.White_Beam.wb_functions import (
                                             convert2EPL,
                                             ellipse,
                                             ideal_ellipse,
                                             plot_ellipse,
                                             plot_widths
                                             )
from general.misc import create_folder
from timeit import default_timer as timer


def norm_jets(prj_fld, dark, flatfield, test_matrix, norm_fld):
    for index, test_name in enumerate(test_matrix['Test']):
        test_path = '{0}/Images/Uniform_Jets/Mean/AVG_{1}'\
                    '.tif'.format(prj_fld, test_name)

        # Offset bounds found in ImageJ, X and Y are flipped!
        sl_x_start = test_matrix['BY'][index]
        sl_x_end = sl_x_start + test_matrix['Height'][index]
        offset_sl_x = slice(sl_x_start, sl_x_end)

        sl_y_start = test_matrix['BX'][index]
        sl_y_end = sl_y_start + test_matrix['Width'][index]
        offset_sl_y = slice(sl_y_start, sl_y_end)

        # Read in averaged images and normalize
        data = np.array(Image.open(test_path))
        data_norm = np.zeros(np.shape(data), dtype=float)
        warnings.filterwarnings('ignore')
        data_norm = (data - dark) / (flatfield - dark)
        warnings.filterwarnings('default')

        # Apply intensity correction
        offset_norm = np.nanmedian(data_norm[offset_sl_x, offset_sl_y]) - 1
        data_norm -= offset_norm

        # Save Transmission images
        im = Image.fromarray(data_norm)
        im.save(norm_fld + '/' + test_path.rsplit('/')[-1].replace('AVG',
                                                                   'Norm'))


def proc_jet(cm_px, save_fld, scint, index, test_name,
             test_path, TtoEPL, EPLtoT, offset_sl_x, offset_sl_y, data_norm):
    # Create folders
    epl_fld = create_folder(save_fld + '/EPL')
    summ_fld = create_folder(save_fld + '/Summary')
    ratio_elps_fld = create_folder(save_fld + '/RatioEllipse')
    ratio_peak_fld = create_folder(save_fld + '/RatioPeak')
    vert_elps_fld = create_folder(save_fld + '/EllipseVertical')
    vert_peak_fld = create_folder(save_fld + '/PeakVertical')
    vert_opt_fld = create_folder(save_fld + '/OpticalVertical')
    ratio_elpsT_fld = create_folder(save_fld + '/RatioEllipseT')
    ratio_peakT_fld = create_folder(save_fld + '/RatioPeakT')
    boundary_folder = create_folder(save_fld + '/Boundaries')
    graph_fld = create_folder(save_fld + '/Graphs/' + test_name)
    width_fld = create_folder(save_fld + '/Widths/' + test_name)
    scans_fld = create_folder(save_fld + '/Scans/')

    # Construct EPL mapping
    data_epl = np.zeros(np.shape(data_norm), dtype=float)
    cropped_view = np.linspace(start=20, stop=325, num=325-20+1, dtype=int)
    for _,k in enumerate(cropped_view):
        data_epl[k, :] = TtoEPL[k](data_norm[k, :])

    # Correct the EPL values
    offset_epl = np.nanmedian(data_epl[offset_sl_x, offset_sl_y])
    data_epl -= offset_epl

    # Rotate the 700 um jet
    if '700' in test_name:
        data_epl = rotate(data_epl, 2.0)

    # Save EPL images
    im = Image.fromarray(data_epl)
    im.save(epl_fld + '/' + test_path.rsplit('/')[-1].replace('AVG', scint))

    left_bound = len(cropped_view) * [np.nan]
    right_bound = len(cropped_view) * [np.nan]
    elps_epl = len(cropped_view) * [np.nan]
    optical_diameter = len(cropped_view) * [np.nan]
    peak_epl = len(cropped_view) * [np.nan]
    axial_position = len(cropped_view) * [np.nan]
    lateral_position = len(cropped_view) * [np.nan]
    fitted_graph = len(cropped_view) * [np.nan]
    epl_graph = len(cropped_view) * [np.nan]
    optical_T = len(cropped_view) * [np.nan]
    peak_T = len(cropped_view) * [np.nan]
    elps_T = len(cropped_view) * [np.nan]

    for z, k in enumerate(cropped_view):
        smoothed = savgol_filter(data_epl[k, :], 105, 7)
        warnings.filterwarnings('ignore')
        peaks, _ = find_peaks(smoothed, width=50, prominence=0.01)
        warnings.filterwarnings('default')

        if len(peaks) == 1:
            # Ellipse fitting
            warnings.filterwarnings('ignore')
            [rel_width, rel_max, lpos, rpos] = peak_widths(
                                                           smoothed,
                                                           peaks,
                                                           rel_height=0.80
                                                           )
            warnings.filterwarnings('default')

            rel_width = rel_width[0]
            rel_max = rel_max[0]
            left_bound[z] = lpos[0]
            right_bound[z] = rpos[0]
            ydata = smoothed[int(round(lpos[0])):int(round(rpos[0]))]

            warnings.filterwarnings('ignore')
            ide_data = ideal_ellipse(ydata, rel_width, rel_max, cm_px)
            elps_epl[z] = ide_data[0]
            fitted_graph[z] = ide_data[1]
            epl_graph[z] = ide_data[2]
            warnings.filterwarnings('default')

            optical_diameter[z] = rel_width * cm_px
            axial_position[z] = k
            lateral_position[z] = int(np.mean([lpos[0], rpos[0]]))
            peak_epl[z] = smoothed[peaks[0]]

            # Convert diameters to transmissions
            optical_T[z] = EPLtoT[k](optical_diameter[z])
            peak_T[z] = EPLtoT[k](peak_epl[z])
            elps_T[z] = EPLtoT[k](elps_epl[z])

            # Plot the fitted and EPL graphs
            if z % 15 == 0:
                # Ellipse
                plot_ellipse(
                             epl_graph[z],
                             fitted_graph[z],
                             '{0}/{1}_{2:03d}.png'.format(graph_fld, scint, k)
                             )
                plt.close()

                plot_widths(
                            data_epl[k, :],
                            peaks,
                            rel_max,
                            lpos[0],
                            rpos[0],
                            '{0}/{1}_{2:03d}.png'.format(width_fld, scint, k)
                            )
                plt.close()

    if len(peaks) == 1:
        lat_start = int(round(np.nanmean(lateral_position)))-20
        lat_end = int(round(np.nanmean(lateral_position)))+20
        signal = np.nanmean(data_epl[20:325, lat_start:lat_end])
    else:
        signal = 0

    noise = np.nanstd(data_epl[offset_sl_x, offset_sl_y])
    SNR = signal / noise

    # Calculate ratios
    ratio_ellipseT = np.array(elps_T) / np.array(optical_T)
    ratio_peakT = np.array(peak_T) / np.array(optical_T)
    ratio_ellipse = np.array(elps_epl) / np.array(optical_diameter)
    ratio_peak = np.array(peak_epl) / np.array(optical_diameter)

    mean_ratio_ellipseT = np.nanmean(ratio_ellipseT)
    mean_ratio_peakT = np.nanmean(ratio_peakT)
    mean_ratio_ellipse = np.nanmean(ratio_ellipse)
    mean_ratio_peak = np.nanmean(ratio_peak)

    cv_ratio_ellipseT = np.nanstd(ratio_ellipseT) / np.nanmean(ratio_ellipseT)
    cv_ratio_peakT = np.nanstd(ratio_peakT) / np.nanmean(ratio_peakT)
    cv_ratio_ellipse = np.nanstd(ratio_ellipse) / np.nanmean(ratio_ellipse)
    cv_ratio_peak = np.nanstd(ratio_peak) / np.nanmean(ratio_peak)

    with np.errstate(all='ignore'):
        ellipse_errors = [
                          rmse(elps_epl, optical_diameter),
                          mape(elps_epl, optical_diameter),
                          zeta(elps_epl, optical_diameter),
                          mdlq(elps_epl, optical_diameter)
                          ]
        peak_errors = [
                       rmse(peak_epl, optical_diameter),
                       mape(peak_epl, optical_diameter),
                       zeta(peak_epl, optical_diameter),
                       mdlq(peak_epl, optical_diameter)
                       ]

    processed_data = {
                      'Diameters': [optical_diameter, elps_epl, peak_epl],
                      'Transmissions': [optical_T, elps_T, peak_T],
                      'Axial Position': axial_position,
                      'Lateral Position': lateral_position,
                      'Bounds': [left_bound, right_bound],
                      'Offset EPL': offset_epl,
                      'SNR': SNR,
                      'Ellipse Errors': ellipse_errors,
                      'Peak Errors': peak_errors,
                      'Transmission Ratios': [ratio_ellipseT, ratio_peakT],
                      'EPL Ratios': [ratio_peak, ratio_ellipse]
                      }

    with open(summ_fld + '/' + scint + '_' + test_name + '.pckl', 'wb') as f:
        pickle.dump(processed_data, f)

    with open('{0}/{1}_{2}_y170.pckl'.format(scans_fld, scint,
                                             test_name), 'wb') as f:
        pickle.dump([data_epl[170, :],
                     savgol_filter(data_epl[170, :], 105, 7)], f)

    with open('{0}/{1}_{2}_y60.pckl'.format(scans_fld, scint,
                                            test_name), 'wb') as f:
        pickle.dump([data_epl[60, :],
                     savgol_filter(data_epl[60, :], 105, 7)], f)

    # Boundary plot
    plt.figure()
    plt.imshow(
               data_epl,
               vmin=0,
               vmax=0.25,
               zorder=1
               )
    plt.scatter(
                x=left_bound,
                y=axial_position,
                s=1,
                color='red',
                zorder=2
                )
    plt.scatter(
                x=right_bound,
                y=axial_position,
                s=1,
                color='red',
                zorder=2
                )
    plt.scatter(
                x=lateral_position,
                y=axial_position,
                s=1,
                color='white',
                zorder=2
                )
    plt.title(test_name)
    plt.savefig('{0}/{1}.png'.format(boundary_folder, test_name))
    plt.close()

    # Vertical ellipse variation
    plt.figure()
    plt.plot(
             axial_position,
             np.array(elps_epl)*10,
             ' o'
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Ellipse Diameter (mm)')
    plt.ylim([0.4, 2.15])
    plt.title(test_name)
    plt.savefig('{0}/{1}_{2}.png'.format(vert_elps_fld, scint, test_name))
    plt.close()

    # Vertical peak variation
    plt.figure()
    plt.plot(
             axial_position,
             np.array(peak_epl)*10,
             ' o'
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Peak Diameter (mm)')
    plt.ylim([0.4, 2.15])
    plt.title(test_name)
    plt.savefig('{0}/{1}_{2}.png'.format(vert_peak_fld, scint, test_name))
    plt.close()

    # Vertical optical variation
    plt.figure()
    plt.plot(
             axial_position,
             np.array(optical_diameter)*10,
             ' o'
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Optical Diameter (mm)')
    plt.ylim([0.4, 2.15])
    plt.title(test_name)
    plt.savefig('{0}/{1}_{2}.png'.format(vert_opt_fld, scint, test_name))
    plt.close()

    # Vertical EPL ratio (ellipse)
    plt.figure()
    plt.plot(
             axial_position,
             ratio_ellipse,
             ' o'
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Ellipse/Optical EPL Ratio')
    plt.ylim([0.4, 1.1])
    plt.savefig('{0}/{1}_{2}.png'.format(ratio_elps_fld, scint, test_name))
    plt.close()

    # Vertical EPL ratio (peak)
    plt.figure()
    plt.plot(
             axial_position,
             ratio_peak,
             ' o'
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Peak/Optical EPL Ratio')
    plt.ylim([0.4, 1.1])
    plt.savefig('{0}/{1}_{2}.png'.format(ratio_peak_fld, scint, test_name))
    plt.close()

    # Vertical transmission ratio (ellipse)
    plt.figure()
    plt.plot(
             axial_position,
             ratio_ellipseT,
             ' o'
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Ellipse/Optical Transmission Ratio')
    plt.ylim([0.8, 1.3])
    plt.title(test_name)
    plt.savefig('{0}/{1}_{2}.png'.format(ratio_elpsT_fld, scint, test_name))
    plt.close()

    # Vertical transmission ratio (peak)
    plt.figure()
    plt.plot(
             axial_position,
             ratio_peakT,
             ' o'
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Peak/Optical Transmission Ratio')
    plt.ylim([0.8, 1.3])
    plt.title(test_name)
    plt.savefig('{0}/{1}_{2}.png'.format(ratio_peakT_fld, scint, test_name))
    plt.close()


def main():
    # Location of APS 2018-1 data
    prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'

    #%% Imaging setup
    cm_px = 0.16 / 162   # See 'APS White Beam.xlsx -> Pixel Size'

    dark_path = prj_fld + '/Images/Uniform_Jets/Mean/AVG_Jet_dark2.tif'
    dark = np.array(Image.open(dark_path))

    flat_path = prj_fld + '/Images/Uniform_Jets/Mean/AVG_Jet_flat2.tif'
    flatfield = np.array(Image.open(flat_path))

    matrix_path = prj_fld + '/APS White Beam.txt'
    test_matrix = pd.read_csv(matrix_path, sep='\t+', engine='python')

    norm_fld = create_folder('{0}/Processed/Normalized/'.format(prj_fld))

    # Normalize the images and save them
    norm_jets(prj_fld, dark, flatfield, test_matrix, norm_fld)

    # Scintillator
    scintillators = ['LuAG', 'YAG']

    for scint in scintillators:
        #%% Load models from the whitebeam_2018-1 script
        f = open(prj_fld + '/Model/water_model_' + scint + '.pckl', 'rb')
        water_mdl = pickle.load(f)
        f.close()

        f = open(prj_fld + '/Model/KI1p6_model_' + scint + '.pckl', 'rb')
        KI1p6_mdl = pickle.load(f)
        f.close()

        f = open(prj_fld + '/Model/KI3p4_model_' + scint + '.pckl', 'rb')
        KI3p4_mdl = pickle.load(f)
        f.close()

        f = open(prj_fld + '/Model/KI4p8_model_' + scint + '.pckl', 'rb')
        KI4p8_mdl = pickle.load(f)
        f.close()

        f = open(prj_fld + '/Model/KI8p0_model_' + scint + '.pckl', 'rb')
        KI8p0_mdl = pickle.load(f)
        f.close()

        f = open(prj_fld + '/Model/KI10p0_model_' + scint + '.pckl', 'rb')
        KI10p0_mdl = pickle.load(f)
        f.close()

        f = open(prj_fld + '/Model/KI11p1_model_' + scint + '.pckl', 'rb')
        KI11p1_mdl = pickle.load(f)
        f.close()

        # Top-level save folder
        save_fld = '{0}/Processed/{1}/'.format(prj_fld, scint)

        KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]
        models = [water_mdl, KI1p6_mdl, KI3p4_mdl, KI4p8_mdl, KI8p0_mdl,
                  KI10p0_mdl, KI11p1_mdl]

        for index, test_name in enumerate(test_matrix['Test']):
        #for index in sl:
            test_path = norm_fld + '/Norm_' + test_name + '.tif'
            model = models[KI_conc.index(test_matrix['KI %'][index])]
            nozzleD = test_matrix['Nozzle Diameter (um)'][index]
            KIperc = test_matrix['KI %'][index]
            TtoEPL = model[0]
            EPLtoT = model[1]

            # Offset bounds found in ImageJ, X and Y are flipped!
            sl_x_start = test_matrix['BY'][index]
            sl_x_end = sl_x_start + test_matrix['Height'][index]
            offset_sl_x = slice(sl_x_start, sl_x_end)

            sl_y_start = test_matrix['BX'][index]
            sl_y_end = sl_y_start + test_matrix['Width'][index]
            offset_sl_y = slice(sl_y_start, sl_y_end)

            # Load in normalized images
            data_norm = np.array(Image.open(test_path))

            # Process the jet file
            proc_jet(cm_px, save_fld, scint, index, test_name, test_path,
                     TtoEPL, EPLtoT, offset_sl_x, offset_sl_y, data_norm)

# Run this script
if __name__ == '__main__':
    main()
