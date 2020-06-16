"""
-*- coding: utf-8 -*-
Re-processes the jets with the correction factors.

@Author: rahmann
@Date:   2020-05-01 19:47:14
@Last Modified by:   rahmann
@Last Modified time: 2020-05-01 19:47:14
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
from jet_processing import proc_jet
from general.misc import create_folder

def main():
    # Location of APS 2018-1 data
    prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'

    #%% Imaging setup
    cm_px = 0.16 / 162   # See 'APS White Beam.xlsx -> Pixel Size'

    test_matrix = pd.read_csv(prj_fld + '/APS White Beam.txt', sep='\t+')

    # Scintillator
    scintillators = ['LuAG', 'YAG']

    # Correction factors
    correction_factors = ['peakT', 'ellipseT']

    for cf_type in correction_factors:
        for scint in scintillators:
            #%% Load models from the whitebeam_2018-1 script
            f = open(prj_fld + '/Model/water_model_' + scint + '.pckl', 'rb')
            water_model = pickle.load(f)
            f.close()

            f = open(prj_fld + '/Model/KI1p6_model_' + scint + '.pckl', 'rb')
            KI1p6_model = pickle.load(f)
            f.close()

            f = open(prj_fld + '/Model/KI3p4_model_' + scint + '.pckl', 'rb')
            KI3p4_model = pickle.load(f)
            f.close()

            f = open(prj_fld + '/Model/KI4p8_model_' + scint + '.pckl', 'rb')
            KI4p8_model = pickle.load(f)
            f.close()

            f = open(prj_fld + '/Model/KI8p0_model_' + scint + '.pckl', 'rb')
            KI8p0_model = pickle.load(f)
            f.close()

            f = open(prj_fld + '/Model/KI10p0_model_' + scint + '.pckl', 'rb')
            KI10p0_model = pickle.load(f)
            f.close()

            f = open(prj_fld + '/Model/KI11p1_model_' + scint + '.pckl', 'rb')
            KI11p1_model = pickle.load(f)
            f.close()

            # Top-level save folder
            save_fld = '{0}/Corrected/{1}_{2}/'\
                       .format(prj_fld, scint, cf_type)

            # Load in corresponding correction factor
            cf = np.loadtxt('{0}/Processed/{1}/{1}_{2}_cf.txt'
                            .format(prj_fld, scint, cf_type))

            KI_conc = [0, 1.6, 3.4, 4.8, 8, 10, 11.1]
            models = [water_model, KI1p6_model, KI3p4_model, KI4p8_model,
                      KI8p0_model, KI10p0_model, KI11p1_model]

            for index, test_name in enumerate(test_matrix['Test']):
                test_path = '{0}/Processed/Normalized/Norm_{1}.tif'\
                            .format(prj_fld, test_name)
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

                # Apply corresponding correction factor to normalized image
                data_norm /= cf[KI_conc.index(test_matrix['KI %'][index])]

                # Process the jet file
                proc_jet(cm_px, save_fld, scint, index, test_name,
                         test_path, TtoEPL, EPLtoT, offset_sl_x,
                         offset_sl_y, data_norm)


# Run this script
if __name__ == '__main__':
    main()
