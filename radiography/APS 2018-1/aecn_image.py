# -*- coding: utf-8 -*-
"""

@author: rahmann
"""


import h5py
import pickle
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d

from monobeam_aecn import horiz_scan
from general.spectrum_modeling import density_KIinH2O
from general.misc import create_folder
from general.calc_statistics import rmse


prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'
hdf5_fld = '{0}/HDF5'.format(prj_fld)
mb_fld = create_folder('{0}/Monobeam/AeroECN'.format(prj_fld))


def main():
    # See 'Spray Imaging' in Excel workbook
    cm_px = np.loadtxt('{0}/cm_px.txt'.format(prj_fld))

    fly_scan = 'Fly_Scans_208_340'
    images = 'center'

    # Injector center position on image (in pixels)
    image_inj_x = 390
    image_inj_y = 18

    # Load in fly scan data
    f = h5py.File('{0}/{1}.hdf5'.format(hdf5_fld, fly_scan), 'r')
    x = np.array(f['X'])
    y = np.array(f['Y'])
    ext_lengths = np.array(f['Radiography'])

    # Attenuation coefficient (total w/o coh. scattering - cm^2/g)
    # Convert to mm^2/g for mm, multiply by density in g/mm^3
    # Pure water @ 8 keV
    # <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
    atten_coeff = (1.006*10*(10*10))*(0.001)

    # Calculate EPL and convert to um from mm
    EPL = (ext_lengths / atten_coeff) * 1000

    # Load in corresponding TimeRes EPL image and convert to um from cm
    img_path = '{0}/Images/AeroECN 100psi 10%KI/EPL/TimeRes/'\
               'AeroECN_centerB_.tif'.format(prj_fld)

    img = np.array(Image.open(img_path))
    img *= 10000

    xx = np.linspace(1, 768, 768)
    xx = xx - image_inj_x[n]
    # Convert to mm
    xx = xx * cm_px * 10

    yy = np.linspace(1, 352, 352)
    yy = yy - image_inj_y[n]
    # Convert to mm
    yy = yy * cm_px * 10
    
    # Image positions
    image_h_x = [0, 767]
    image_h_y = list(range(29, 330))

    # Initialize the image and fly scan arrays
    inj_xdata = len(image_h_y) * [None]
    inj_ydata = len(image_h_y) * [None]
    img_xdata = len(image_h_y) * [None]
    img_ydata = len(image_h_y) * [None]
    h_y = len(image_h_y) * [None]

    for j in image_h_y:
        # Get horizontal scan data (H1)
        [inj_xdata[j], inj_ydata[j], img_xdata[j], img_ydata[j], _h_y[j]] = \
            horiz_scan(xx, image_h_x, yy, j, img, EPL, x, y)
    
    plt.figure()
    plt.contourf([xx, yy[image_h_y]], img_ydata)
    plt.savefig('{0}/image.png'.format(mb_fld))
    plt.close()


# Run this script
if __name__ == '__main__':
    main()
