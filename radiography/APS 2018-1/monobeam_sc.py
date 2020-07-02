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

from general.calc_statistics import rmse
from general.spectrum_modeling import density_KIinH2O
from general.misc import create_folder
from monobeam_aecn import plot_fig


prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'
hdf5_fld = '{0}/HDF5'.format(prj_fld)
mb_fld = create_folder('{0}/Monobeam/SC'.format(prj_fld))


def horiz_scan(xx, image_h_x, yy, image_h_y, img, EPL, x, y):
    # Get horizontal scan indices
    h_y = yy[image_h_y]
    h_x1 = xx[image_h_x[0]]
    h_x2 = xx[image_h_x[1]]
    h_xslice = list(range(
                          np.argmin(abs(h_x1 - x)),
                          np.argmin(abs(h_x2 - x)) + 1
                          )
                    )
    h_yslice = np.argmin(np.abs(h_y - np.mean(y, axis=0)))

    # Injector data
    inj_xdata = [x[z][0] for z in h_xslice]
    inj_ydata = [EPL[z] for z in h_xslice]

    # Get horizontal scan image data for plots
    img_xdata = xx[image_h_x[0]:image_h_x[1]]
    img_ydata = img[image_h_y, image_h_x[0]:image_h_x[1]]

    return inj_xdata, inj_ydata, img_xdata, img_ydata, h_y


def plot_fig(img_x, img_y, inj_x, inj_y, xlabel, image_h_y, h_y, name):
    plt.figure()
    plt.plot(
             img_x,
             img_y,
             label='WB @ {0} px'.format(image_h_y),
             linestyle='solid',
             linewidth=2.0,
             color='k'
             )
    plt.plot(
             inj_x,
             inj_y,
             label='MB @ {0:0.2f} mm'.format(h_y),
             linestyle='dashed',
             linewidth=2.0,
             color='b'
             )
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('EPL ($\mu$m)')
    plt.title('{0}: y = {1:0.2f} mm'.format(name, h_y))
    plt.savefig('{0}/{1}_{2:0.2f}.png'.format(mb_fld, name, h_y))
    plt.close()

    return


def main():
    # See 'Spray Imaging' in Excel workbook
    cm_px = np.loadtxt('{0}/cm_px.txt'.format(prj_fld))

    # Injector center position on image (in pixels)
    image_inj_x = [471, 354]
    image_inj_y = [22, -190]

    # Image positions (see 'WB & MB Python' in Excel workbook)
    image_h_x = [0, 767]
    image_h_y = [[234], [234, 22]]

    upper_mb = list(range(644, 661+1))
    lower_mb = list(range(662, 679+1))
    images = ['upper', 'lower']

    # Initialize error array
    errors = len(upper_mb) * [None]

    upper_mb = [upper_mb[1]]
    lower_mb = [lower_mb[1]]
    for p, _ in enumerate(upper_mb):
        scans = [upper_mb[p], lower_mb[p]]
        errors[p] = np.zeros(2)
        for n, scan in enumerate(scans):
            # Load in scan data
            f = h5py.File('{0}/Scan_{1}.hdf5'.format(hdf5_fld, scan), 'r')
            x = np.array(f['X'])
            y = np.array(f['Y'])
            theta = np.array(f['Theta'])
            BIM = np.array(f['BIM'])
            PIN = np.array(f['PINDiode'])
            extinction_length = np.log(BIM / PIN)
            offset = np.median(extinction_length[0:10])
            extinction_length -= offset

            # Attenuation coefficient (total w/o coh. scattering - cm^2/g)
            # Convert to mm^2/g for mm, multiply by density in g/mm^3
            # Pure water @ 8 keV
            # <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
            atten_coeff = (1.006*10*(10*10))*(0.001)

            # Calculate EPL and convert to um from mm
            EPL = (extinction_length / atten_coeff) * 1000
            EPL = np.mean(EPL, axis=1)

            # Load in corresponding EPL image and convert to um from cm
            img_path = '{0}/Images/SC/EPL/TimeAvg/'\
                       'AVG_AeroECN_0p3gpm_{1}A_.tif'.format(prj_fld,
                                                             images[n])
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

            for q, m in enumerate(image_h_y[n]):
                # Get horizontal scan data
                [inj_xdata, inj_ydata, img_xdata, img_ydata, h_y] = \
                    horiz_scan(xx, image_h_x, yy, m, img, EPL, x, y)

                # Plot the WB/MB comparison
                plot_fig(img_xdata, img_ydata, inj_xdata, inj_ydata,
                         'Horiz. Location (mm)', m, h_y,
                         '{0}'.format(images[n]))

                if q == 0:
                    # Find a characteristic error for the y = 234 px case
                    interp = interp1d(inj_xdata, inj_ydata, fill_value=0,
                                      bounds_error=False)
                    errors[p][n] = rmse(img_ydata, interp(img_xdata))

    #mean_errors = [np.mean(x) for x in errors]


# Run this script
if __name__ == '__main__':
    main()
