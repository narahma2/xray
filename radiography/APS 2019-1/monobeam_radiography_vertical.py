# -*- coding: utf-8 -*-
"""
Created on 31 May 2020

@author: rahmann
"""


import h5py
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter
from general.spectrum_modeling import density_KIinH2O
from general.misc import create_folder


def main():
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'
    test_conditions = pd.read_csv('{0}/test_conditions.txt'.format(prj_fld),
                                  sep='\t+', engine='python')
    grouped = test_conditions.groupby(by=['Imaging'])
    tests_keys = list(grouped.groups.keys())[4:]
    hdf5_fld = '{0}/HDF5'.format(prj_fld)

    cm_px = 0.05 / 52   # See 'Pixel Size' in Excel workbook

    # Load offsets from the monobeam_radiography script
    offset = np.median(np.loadtxt(prj_fld + '/offsets.txt'))

    for z in tests_keys:
        indices = [
                   i
                   for i, s in enumerate(list(grouped.get_group(z)
                                              ['X Positions (mm)']
                                              )
                                        )
                   if 'to' not in s
                   ]
        temp_list = list(grouped.get_group(z)['Scan'])
        EPL = len(indices) * [None]
        wbp = len(indices) * [None]
        xp = len(indices) * [None]
        scan = len(indices) * [None]
        fy = len(indices) * [None]

        # Load in corresponding EPL image and convert to um from cm
        img_path = '{0}/Images/Spray/EPL/TimeAvg/AVG_{1}_S001.tif'\
                   .format(prj_fld, z)
        img_name = img_path.rsplit('/')[-1].rsplit('_')[0]
        img = np.array(Image.open(img_path))
        img *= 10000

        # Flip image to match MB scan
        img = np.fliplr(img)

        # Find middle of the spray (should be around 768/2=384)
        spray_middle = np.zeros((512,1), dtype=float)
        for i, m in enumerate(img):
            if i >= 35:
                filtered = savgol_filter(m[100:-10], 25, 3)
                spray_middle[i] = np.argmax(filtered) + 99

        mdpt = int(round(np.mean(spray_middle[35:])))

        # Load in the relevant HDF5 scans
        for n, y in enumerate(indices):
            scan[n] = temp_list[y]
            f = h5py.File('{0}/Scan_{1}.hdf5'.format(hdf5_fld, scan[n]), 'r')
            fy[n] = np.array(f['Y'])
            BIM = np.array(f['BIM'])
            PIN = np.array(f['PINDiode'])
            extinction_length = np.log(BIM / PIN)
            extinction_length -= offset

            wbp[n] = round(float(list(grouped.get_group(z)
                           ['X Positions (mm)'])[y]) / (cm_px * 10)) + mdpt
            xp[n] = float(list(grouped.get_group(z)['X Positions (mm)'])[y])

            # Attenuation coefficient (total w/o coh. scattering - cm^2/g)
            # Convert to mm^2/g for mm, multiply by density in g/mm^3
            if 'KI' not in z:
                # Pure water @ 8 keV
                # <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
                atten_coeff = (1.006*10*(10*10))*(0.001)
            elif 'KI' in z:
                # 4.8% KI in water @ 8 keV
                # <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
                atten_coeff = (2.182*10*(10*10))*(density_KIinH2O(4.8)/1000)

            # Calculate EPL and convert to um from mm
            EPL[n] = (extinction_length / atten_coeff) * 1000

        yy = np.linspace(0,511,512)
        yy *= cm_px * 10       # in mm

        mb_fld = create_folder('{0}/Monobeam/{1}'.format(prj_fld, z))

        for n, _ in enumerate(indices):
            plt.figure()
            plt.plot(
                     yy[0::5],
                     img[:, wbp[n]][0::5],
                     color='b',
                     marker='o',
                     fillstyle='none',
                     label='WB {0} @ {1}'.format(img_name, wbp[n])
                     )
            plt.plot(
                     fy[n],
                     np.mean(EPL[n], axis=1),
                     color='r',
                     marker='s',
                     fillstyle='none',
                     label='MB {0}'.format(scan[n])
                     )
            plt.xlim([0, 5])
            plt.ylim([-200, 3500])
            plt.title('Time Averaged - {0} mm'.format(xp[n]))
            plt.xlabel('Vertical Location (mm)')
            plt.ylabel('EPL ($\mu$m)')
            plt.legend()
            plt.savefig('{0}/comparison_x{1}mm.png'.format(mb_fld, xp[n]))
            plt.close()

        plt.figure()
        plt.imshow(img, vmin=-200, vmax=3500)
        plt.colorbar()
        plt.title('EPL ($\mu$m) Mapping of Spray')
        for n, _ in enumerate(indices):
            plt.plot(
                     np.linspace(wbp[n], wbp[n], 512),
                     np.linspace(1,512,512),
                     label='{0} MB'.format(scan[n])
                     )
        plt.savefig('{0}/Spray Image Vertical.png'.format(mb_fld))
        plt.close()

        with open('{0}/monobeam_data_vert.pckl'.format(mb_fld), 'wb') as f:
            pickle.dump([EPL, scan, wbp, xp], f)


# Run this script
if __name__ == '__main__':
    main()
