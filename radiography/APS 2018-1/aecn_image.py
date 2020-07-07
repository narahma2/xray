# -*- coding: utf-8 -*-
"""

@author: rahmann
"""


import glob
import h5py
import pickle
import numpy as np

from PIL import Image

from monobeam_aecn import horiz_scan
from general.misc import create_folder


prj_fld = '/mnt/r/X-ray Radiography/APS 2018-1/'
hdf5_fld = '{0}/HDF5'.format(prj_fld)
mb_fld = create_folder('{0}/Monobeam/AeroECN'.format(prj_fld))


def main():
    # See 'Spray Imaging' in Excel workbook
    cm_px = np.loadtxt('{0}/cm_px.txt'.format(prj_fld))

    fly_scan = 'Fly_Scans_208_340'

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

    # Set the image types in load
    img_types = ['TimeAvg', 'TimeRes']

    for img_type in img_types:
        # Load in corresponding EPL image and convert to um from cm
        img_path = glob.glob(
                             '{0}/Images/*AeroECN*/EPL/{1}/*centerB*'
                             .format(prj_fld, img_type)
                             )[0]

        img = np.array(Image.open(img_path))
        img *= 10000

        xx = np.linspace(1, 768, 768)
        xx = xx - image_inj_x
        # Convert to mm
        xx = xx * cm_px * 10

        yy = np.linspace(1, 352, 352)
        yy = yy - image_inj_y
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

        for i, j in enumerate(image_h_y):
            # Get horizontal scan data (H1)
            [inj_xdata[i], inj_ydata[i],
             img_xdata[i], img_ydata[i], h_y[i]] = horiz_scan(xx, image_h_x,
                                                              yy, j, img, EPL,
                                                              x, y)

        # Save the data sets
        with open('{0}/{1}_aecn.pckl'.format(mb_fld, img_type), 'wb') as f:
            pickle.dump([inj_xdata, inj_ydata, img_xdata, img_ydata,
                         h_y, xx, yy, image_h_y], f)


# Run this script
if __name__ == '__main__':
    main()
