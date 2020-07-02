# -*- coding: utf-8 -*-
"""

@author: rahmann
"""


import pickle
import glob
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from os.path import split
from PIL import Image
from scipy.signal import (
                          savgol_filter,
                          find_peaks,
                          peak_widths
                          )
from skimage.transform import rotate
from general.calc_statistics import (
                                     rmse,
                                     mape,
                                     zeta,
                                     mdlq
                                     )
from general.wb_functions import (
                                  convert2EPL,
                                  ellipse,
                                  ideal_ellipse,
                                  plot_ellipse,
                                  plot_widths
                                  )
from general.misc import create_folder


def norm_jets(prj_fld, dark, flatfield, test_matrix, norm_fld):
    # Look at only the Calibration2 files (NewYAG)
    calib2_files = glob.glob('{0}/Images/Calibration/Mean/'
                             '*Calibration2*'.format(prj_fld))

    # Add KI to the glob
    ki_files = glob.glob('{0}/Images/Calibration/Mean/*KI*'.format(prj_fld))
    calib2_files.append(ki_files[0])

    # Get Calibration2 file names to look up in matrix table
    calib2_names = [
                    split(x)[1].rsplit('.')[0].rsplit('AVG_')[1]
                    for x in calib2_files
                    ]

    for index, test_name in enumerate(test_matrix['Test']):
        if test_name in calib2_names:
            test_path = '{0}/Images/Calibration/Mean/AVG_{1}'\
                        '.tif'.format(prj_fld, test_name)

            # Load in offset file (found after the fact in ImageJ)
            if 'KI' in test_name:
                ofst = np.loadtxt('{0}/Processed/offset_KI.csv'
                                  .format(prj_fld),
                                  skiprows=1,
                                  delimiter=',')
            else:
                ofst = np.loadtxt('{0}/Processed/offset_water.csv'
                                  .format(prj_fld),
                                  skiprows=1,
                                  delimiter=',')
            # Get only the Y values and calculate offset from 1
            ofst = ofst[:,1] - 1

            # Read in averaged images and normalize
            data = np.array(Image.open(test_path))
            data_norm = np.zeros(np.shape(data), dtype=float)
            warnings.filterwarnings('ignore')
            data_norm = (data - dark) / (flatfield - dark)
            warnings.filterwarnings('default')

            # Apply offset
            data_norm = data_norm.T - ofst
            data_norm = data_norm.T

            # Save Transmission images
            im = Image.fromarray(data_norm)
            im.save(norm_fld + '/' + split(test_path)[1].replace('AVG',
                                                                 'Norm'))


def proc_jet(cropped_view, cm_px, save_fld, scint, index, test_name,
             test_path,TtoEPL,EPLtoT,offset_sl_x,offset_sl_y,data_norm,iface):
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
    bnd_fld = create_folder(save_fld + '/Boundaries')
    graph_fld = create_folder(save_fld + '/Graphs/' + test_name)
    width_fld = create_folder(save_fld + '/Widths/' + test_name)
    scans_fld = create_folder(save_fld + '/Scans/')

    # Construct EPL mapping
    data_epl = np.zeros(np.shape(data_norm), dtype=float)
    ofst_epl = np.zeros(np.shape(data_norm)[0], dtype=float)
    for _,k in enumerate(cropped_view):
        data_epl[k, :] = TtoEPL[k](data_norm[k, :])

    # Correct the EPL values
#    ofst_epl = np.nanmedian(data_epl[offset_sl_x, offset_sl_y])
#    data_epl -= ofst_epl

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
        smoothed = savgol_filter(data_epl[k, :], 5, 3)
        warnings.filterwarnings('ignore')
        peaks, _ = find_peaks(smoothed, width=100)
        warnings.filterwarnings('default')

        if len(peaks) == 1:
            # Offset EPL calculation (row-wise)
            warnings.filterwarnings('ignore')
            [_, _, lpos, rpos] = peak_widths(
                                             smoothed,
                                             peaks,
                                             rel_height=0.85
                                             )
            warnings.filterwarnings('default')

            left_side = data_epl[k, 0:int(round(lpos[0]))-10]
            right_side = data_epl[k, int(round(rpos[0]))+10:-1]
            ofst_epl[k] = np.mean([np.nanmedian(left_side),
                                   np.nanmedian(right_side)])
            data_epl[k, :] -= ofst_epl[k]
            #if '005' in test_name:
            #    breakpoint()
            smoothed -= ofst_epl[k]

            # Ellipse fitting
            warnings.filterwarnings('ignore')
            [rel_width, rel_max, lpos, rpos] = peak_widths(
                                                           smoothed,
                                                           peaks,
                                                           rel_height=0.85
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

    # Save EPL images
    im = Image.fromarray(data_epl)
    im.save(epl_fld + '/' + test_path.rsplit('/')[-1].replace('Norm', 'EPL'))

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
                      'Offset EPL': ofst_epl,
                      'SNR': SNR,
                      'Ellipse Errors': ellipse_errors,
                      'Peak Errors': peak_errors,
                      'Transmission Ratios': [ratio_ellipseT, ratio_peakT],
                      'EPL Ratios': [ratio_peak, ratio_ellipse],
                      'Injector Face': iface
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
               vmax=0.50,
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
    plt.savefig('{0}/{1}.png'.format(bnd_fld, test_name))
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
    plt.xlim([0, 480])
    plt.ylim([0.4, 5.0])
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
    plt.xlim([0, 480])
    plt.ylim([0.4, 5.0])
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
    plt.xlim([0, 480])
    plt.ylim([0.4, 5.0])
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
    plt.xlim([0, 480])
    plt.ylim([0.4, 2.1])
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
    plt.xlim([0, 480])
    plt.ylim([0.4, 2.1])
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
    plt.xlim([0, 480])
    plt.ylim([0.8, 2.3])
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
    plt.xlim([0, 480])
    plt.ylim([0.8, 2.3])
    plt.title(test_name)
    plt.savefig('{0}/{1}_{2}.png'.format(ratio_peakT_fld, scint, test_name))
    plt.close()


def main():
    # Location of APS 2018-1 data
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'

    #%% Imaging setup
    cm_px = 0.05 / 52   # See 'Pixel Size' in Excel workbook

    dark_path = prj_fld + '/Images/Flat/AVG_dark_current.tif'
    dark = np.array(Image.open(dark_path))

    flat_path = prj_fld + '/Images/Flat/Mean/AVG_Background_NewYAG.tif'
    flatfield = np.array(Image.open(flat_path))

    matrix_path = prj_fld + '/test_matrix.txt'
    test_matrix = pd.read_csv(matrix_path, sep='\t+', engine='python')

    # Create folder for the Normalized sets
    norm_fld = create_folder('{0}/Processed/Normalized/'.format(prj_fld))

    # Normalize the all of the images and save them
    norm_jets(prj_fld, dark, flatfield, test_matrix, norm_fld)

    # Scintillator used
    scint = 'YAG'

    #%% Load models from the whitebeam_2019 script
    f = open(prj_fld + '/Model/water_model_' + scint + '.pckl', 'rb')
    water_mdl = pickle.load(f)
    f.close()

    f = open(prj_fld + '/Model/KI4p8_model_' + scint + '.pckl', 'rb')
    KI4p8_mdl = pickle.load(f)
    f.close()

    # Top-level save folder
    save_fld = '{0}/Processed/{1}/'.format(prj_fld, scint)

    KI_conc = [0, 4.8]
    models = [water_mdl, KI4p8_mdl]

    # Look at only the Calibration2 files (NewYAG)
    calib2_files = glob.glob('{0}/Images/Calibration/Mean/'
                             '*Calibration2*'.format(prj_fld))

    # Add KI to the glob
    ki_files = glob.glob('{0}/Images/Calibration/Mean/*KI*'.format(prj_fld))
    calib2_files.append(ki_files[0])

    # Get Calibration2 file names to look up in matrix table
    calib2_names = [
                    split(x)[1].rsplit('.')[0].rsplit('AVG_')[1]
                    for x in calib2_files
                    ]

    for index, test_name in enumerate(test_matrix['Test']):
        if test_name in calib2_names:
            test_path = norm_fld + '/Norm_' + test_name + '.tif'

            # Load relevant model
            if 'KI' in test_name:
                model = models[1]
            else:
                model = models[0]
            TtoEPL = model[0]
            EPLtoT = model[1]

            # Cropping window
            crop_start = test_matrix['Cropping Start'][index]
            crop_stop = 480
            cropped_view = np.linspace(
                                       start=crop_start,
                                       stop=crop_stop,
                                       num=crop_stop-crop_start+1, dtype=int
                                       )

            # ROI for offset calculations
            roi = test_matrix['ROI'][index]
            sl_x_start = int(roi.rsplit(',')[0].rsplit(':')[0][1:])
            sl_x_end = int(roi.rsplit(',')[0].rsplit(':')[1])
            offset_sl_x = slice(sl_x_start, sl_x_end)

            sl_y_start = int(roi.rsplit(",")[1].rsplit(":")[0])
            sl_y_end = int(roi.rsplit(",")[1].rsplit(":")[1][:-1])
            offset_sl_y = slice(sl_y_start, sl_y_end)

            # Injector face
            iface = test_matrix['Injector Face'][index]

            # Load in normalized images
            data_norm = np.array(Image.open(test_path))

            # Process the jet file
            proc_jet(cropped_view, cm_px, save_fld, scint, index, test_name,
                     test_path, TtoEPL, EPLtoT, offset_sl_x, offset_sl_y,
                     data_norm, iface)

# Run this script
if __name__ == '__main__':
    main()

