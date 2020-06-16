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


# Location of APS 2019-1 data
prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'


def norm_spray(test_path, flat_path, dark_path, save_fld):
    data = np.array(Image.open(test_path))
    flat = np.array(Image.open(flat_path))
    dark = np.array(Image.open(dark_path))
    data_norm = (data - dark) / (flat - dark)

    # Calculate and apply offset
    ofst = np.nanmedian(data_norm[109:109+134, 659:659+90]) - 1
    data_norm -= ofst

    # Save Transmission images
    im = Image.fromarray(data_norm)
    im.save(save_fld + '/' + split(test_path)[1])


def conv_spray(test_path, save_fld):
    # Load models from the whitebeam_2019 script
    with open(prj_fld + '/Model/water_model_YAG.pckl', 'rb') as f:
        water_mdl = pickle.load(f)

    with open(prj_fld + '/Model/KI4p8_model_YAG.pckl', 'rb') as f:
        KI4p8_mdl = pickle.load(f)

    models = [water_mdl, KI4p8_mdl]

    # Load in correction factors (only use elpsT, worked best in 2018-1)
    # 0: Water elps, 1: Water peak, 2: KI elps, 3: KI peak
    cf_summ = np.loadtxt('{0}/Processed/YAG/Summary/cf_summary.txt'
                         .format(prj_fld),
                         delimiter='\t',
                         skiprows=1
                         )

    # Load corresponding correction factor
    if 'KI' in test_path:
        model = models[1]
        cf = cf_summ[2]
    else:
        model = models[0]
        cf = cf_summ[0]

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
    ofst = np.nanmedian(data_epl[109:109+134, 659:659+90])
    data_epl -= ofst

    # Save EPL image
    im = Image.fromarray(data_epl)
    im.save(save_fld + '/' + split(test_path)[1])


def main():
    # Location of spray images
    spry_fld = '{0}/Images/Spray/'.format(prj_fld)

    # Flat and dark image paths
    flat = '{0}/Images/Flat/Mean/AVG_Background_NewYAG.tif'.format(prj_fld)
    dark = '{0}/Images/Flat/AVG_dark_current.tif'.format(prj_fld)

    # Time-averaged images
    tavg_files = glob.glob('{0}/Mean/*.tif'.format(spry_fld))

    # Time-resolved images (load only the 950th images of the NewYAG set
    tres_fld = glob.glob('{0}/Raw/*'.format(spry_fld))
    tres_files = [
                  glob.glob('{0}/*.tif'.format(x))[950]
                  for x in tres_fld
                  if 'NewYAG' in x
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
