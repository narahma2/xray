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
from calibration_processing import proc_jet


def main():
    # Location of APS 2019-1 data
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'

    # Location of Normalized sets
    norm_fld = '{0}/Processed/Normalized/'.format(prj_fld)

    #%% Imaging setup
    cm_px = 0.05 / 52   # See 'Pixel Size' in Excel workbook

    matrix_path = prj_fld + '/test_matrix.txt'
    test_matrix = pd.read_csv(matrix_path, sep='\t+', engine='python')

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
    save_fld = '{0}/Corrected/{1}/'.format(prj_fld, scint)

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

    # Load in correction factors (only use elpsT, worked best in 2018-1)
    # 0: Water elps, 1: Water peak, 2: KI elps, 3: KI water
    cf_summ = np.loadtxt('{0}/Processed/YAG/Summary/cf_summary.txt'
                         .format(prj_fld),
                         delimiter='\t',
                         skiprows=1
                         )

    for index, test_name in enumerate(test_matrix['Test']):
        if test_name in calib2_names:
            test_path = norm_fld + '/Norm_' + test_name + '.tif'

            # Load relevant model
            if 'KI' in test_name:
                model = models[1]
                cf = cf_summ[2]
            else:
                model = models[0]
                cf = cf_summ[0]

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

            # Apply correction factor
            data_norm /= cf

            # Process the jet file
            proc_jet(cropped_view, cm_px, save_fld, scint, index, test_name,
                     test_path, TtoEPL, EPLtoT, offset_sl_x, offset_sl_y,
                     data_norm, iface)

# Run this script
if __name__ == '__main__':
    main()
