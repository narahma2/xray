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

        # Load in corresponding EPL image and convert to um from cm
        img_path = '{0}/Images/AeroECN 100psi 10%KI/EPL/TimeAvg/'\
                   'AVG_AeroECN_{1}B_.tif'\
                   .format(prj_fld, image)
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

        # Get vertical scan data (V1)
        [inj_xdata, inj_ydata, img_xdata, img_ydata, v_x] = \
            vert_scan(xx, image_v_x[n][0], yy, image_v_y, img, EPL, x, y)
        plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                 'Vertical Location (mm)', 'x', v_x, '{0}_v1'.format(image))

        # Get vertical scan data (V2)
        [inj_xdata, inj_ydata, img_xdata, img_ydata, v_x] = \
            vert_scan(xx, image_v_x[n][1], yy, image_v_y, img, EPL, x, y)
        plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                 'Vertical Location (mm)', 'x', v_x, '{0}_v2'.format(image))

        if n == 0:
            u = 60-29
            v = 150-29
        else:
            u = 0
            v = 1

        # Get horizontal scan data (H1)
        [inj_xdata, inj_ydata, img_xdata, img_ydata, h_y] = \
            horiz_scan(xx, image_h_x, yy, image_h_y[n][u], img, EPL, x, y)
        plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                 'Horizontal Location (mm)', 'y', h_y, '{0}_h1'.format(image))

        # Get horizontal scan data (H2)
        [inj_xdata, inj_ydata, img_xdata, img_ydata, h_y] = \
            horiz_scan(xx, image_h_x, yy, image_h_y[n][v], img, EPL, x, y)
        plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                 'Horizontal Location (mm)', 'y', h_y, '{0}_h2'.format(image))

        # Build up RMSE array for the horizontal center scan
        if n == 0:
            center_rmse = np.zeros(len(image_h_y[n]))
            image_pos = np.zeros(len(image_h_y[n]))
            scan_pos = np.zeros(len(image_h_y[n]))

            for q, k in enumerate(image_h_y[n]):
                [inj_xdata, inj_ydata, img_xdata, img_ydata, h_y] = \
                    horiz_scan(xx, image_h_x, yy, k, img, EPL, x, y)
                interp = interp1d(inj_xdata, inj_ydata, fill_value=0,
                                  bounds_error=False)
                center_rmse[q] = rmse(img_ydata, interp(img_xdata))
                image_pos[q] = k
                scan_pos[q] = h_y

        # Plot the RMSE curve
        plt.figure()
        plt.plot(
                 image_pos,
                 center_rmse,
                 linewidth=2.0,
                 linestyle='solid',
                 color='k'
                 )
        plt.xlabel('Vertical Location (px)')
        plt.ylabel('Norm. RMSE (-)')
        plt.title('RMSE for AeroECN @ 11.1% KI')
        plt.savefig('{0}/rmse.png'.format(mb_fld))
        plt.close()

# Run this script
if __name__ == '__main__':
    main()
