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


# Location of APS 2019-1 data
prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'


def norm_spray(test_path, flat_path, dark_path, save_fld):
    data = np.array(Image.open(test_path))
    flat = np.array(Image.open(flat_path))
    dark = np.array(Image.open(dark_path))
    data_norm = (data - dark) / (flat - dark)

    # Calculate and apply offset
    ofst = np.nanmedian(data_norm[43:43+93, 15:15+94]) - 1
    data_norm -= ofst

    # Save Transmission images
    im = Image.fromarray(data_norm)
    im.save(save_fld + '/' + split(test_path)[1])


def conv_spray(test_path, save_fld):
    # Set scintillator
    scint = 'LuAG'

    # Load models from the whitebeam_2019 script
    with open('{0}/Model/KI10p0_model_{1}.pckl'.format(prj_fld, scint),
              'rb') as f:
        sc_mdl = pickle.load(f)

    with open('{0}/Model/KI11p1_model_{1}.pckl'.format(prj_fld, scint),
              'rb') as f:
        aecn_mdl = pickle.load(f)

    models = [sc_mdl, aecn_mdl]

    # Load in correction factors (only use elpsT, worked best in 2018-1)
    # 0: Water elps, 1: Water peak, 2: KI elps, 3: KI peak
    cf_summ = np.loadtxt('{0}/Processed/{1}/{1}_elpsT_cf.txt'
                         .format(prj_fld, scint),
                         )

    # Load corresponding correction factor and set rotation angle
    if 'SC' in test_path:
        model = models[0]
        cf = cf_summ[5]
        rot = -1.5
    else:
        model = models[1]
        cf = cf_summ[6]
        rot = -1

    TtoEPL = model[0]

    # Load in normalized image
    data_norm = np.array(Image.open(test_path))

    # Apply correction factor
    data_norm /= cf

    # Convert to EPL
    data_epl = np.zeros(np.shape(data_norm), dtype=float)
    for k,_ in enumerate(data_norm):
        data_epl[k,:] = TtoEPL[k](data_norm[k,:])

    # Calculate and apply offset
    ofst = np.nanmedian(data_epl[43:43+93, 15:15+94])
    data_epl -= ofst

    # Rotate image before saving
    data_epl = rotate(data_epl, rot)

    # Save EPL image
    im = Image.fromarray(data_epl)
    im.save(save_fld + '/' + split(test_path)[1])


def main():
    # Spray type
    sprays = ['SC', 'AeroECN 100psi 10%KI']

    for spray in sprays:
        # Location of spray images
        spry_fld = '{0}/Images/{1}/'.format(prj_fld, spray)

        # Flat and dark image paths
        flat = glob.glob('{0}/Mean/*flat*'.format(spry_fld))[0]
        dark = glob.glob('{0}/Mean/*dark*'.format(spry_fld))[0]

        # Time-averaged images
        tavg_files = glob.glob('{0}/Mean/*.tif'.format(spry_fld))

        # Time-resolved images (load only the 950th images of the NewYAG set
        tres_fld = glob.glob('{0}/Raw/*'.format(spry_fld))

        # Remove flat/dark from the folder
        tres_fld = [x for x in tres_fld if 'dark' not in x]
        tres_fld = [x for x in tres_fld if 'flat' not in x]

        tres_files = [
                      glob.glob('{0}/*.tif'.format(x))[500]
                      for x in tres_fld
                      ]

        # Run time-averaged normalization 
        norm_tavg_fld = create_folder('{0}/Norm/TimeAvg/'.format(spry_fld))
        [norm_spray(x, flat, dark, norm_tavg_fld) for x in tavg_files]

        # Run time-resolved normalization
        norm_tres_fld = create_folder('{0}/Norm/TimeRes/'.format(spry_fld))
        [norm_spray(x, flat, dark, norm_tres_fld) for x in tres_files]

        # Run time-averaged EPL conversion
        epl_tavg_fld = create_folder('{0}/EPL/TimeAvg/'.format(spry_fld))
        tavg_files = glob.glob('{0}/*.tif'.format(norm_tavg_fld))
        [conv_spray(x, epl_tavg_fld) for x in tavg_files]

        # Run time-resolved EPL conversion
        epl_tres_fld = create_folder('{0}/EPL/TimeRes/'.format(spry_fld))
        tres_files = glob.glob('{0}/*.tif'.format(norm_tres_fld))
        [conv_spray(x, epl_tres_fld) for x in tres_files]


# Run this script
if __name__ == '__main__':
    main()
