# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:00:32 2019

@author: rahmann
"""


import h5py
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from general.Spectra.spectrum_modeling import density_KIinH2O
from general.misc import create_folder


def main():
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'
    test_conditions = pd.read_csv('{0}/test_conditions.txt'.format(prj_fld),
                                  sep='\t+', engine='python')
    grouped = test_conditions.groupby(by=['Imaging'])
    tests_keys = list(grouped.groups.keys())[3:]

    hdf5_fld = '{0}/HDF5'.format(prj_fld)

    cm_px = 0.05 / 52   # See 'Pixel Size' in Excel workbook

    # Create offsets file
    open(prj_fld + '/offsets.txt', 'w').close()

    for z in tests_keys:
        indices = [
                   i
                   for i, s in enumerate(list(grouped.get_group(z)
                                              ['Y Positions (mm)']
                                              )
                                        )
                   if 'to' not in s
                   ]
        temp_list = list(grouped.get_group(z)['Scan'])
        EPL = len(indices) * [None]
        wbp = len(indices) * [None]
        yp = len(indices) * [None]
        scan = len(indices) * [None]
        x = len(indices) * [None]
        test = z.rsplit('_')[0]

        # Load in the relevant HDF5 scans
        for n, y in enumerate(indices):
            scan[n] = temp_list[y]
            f = h5py.File('{0}/Scan_{1}.hdf5'.format(hdf5_fld, scan[n]), 'r')
            x[n] = np.array(f['X'])
            BIM = np.array(f['BIM'])
            PIN = np.array(f['PINDiode'])
            extinction_length = np.log(BIM / PIN)
            offset = np.median(extinction_length[0:10])
            extinction_length -= offset

            wbp[n] = round(float(list(grouped.get_group(z)
                           ['Y Positions (mm)'])[y]) / (cm_px * 10))
            yp[n] = float(list(grouped.get_group(z)['Y Positions (mm)'])[y])

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

            # Add in air attenuation coefficient
            #atten_coeff *= (9.227*(10*10))*(0.0012929/1000)

            EPL[n] = extinction_length / atten_coeff

            # Append offset value to file
            with open(prj_fld + '/offsets.txt', 'a') as f:
                f.write('{0}\n'.format(offset))

        # Load in corresponding EPL image and convert to mm from cm
        img_path = '{0}/Images/Spray/EPL/TimeAvg/AVG_{1}_S001.tif'\
                   .format(prj_fld, z)
        img = np.array(Image.open(img_path))
        img *= 10

        # Flip image to match MB scan
        img = np.fliplr(img)

        xx = np.linspace(1,768,768)
        xx = xx * cm_px * 10       # in mm
        xx = xx - np.mean(xx)

        mb_fld = create_folder('{0}/Monobeam/{1}'.format(prj_fld, z))

        for n, _ in enumerate(indices):
            plt.figure()
            plt.plot(x[n], np.mean(EPL[n], axis=1), label='Monobeam Averaged')
            plt.plot(xx, img[wbp[n], :], label='Whitebeam Averaged')
            plt.xlim([-3, 3])
            plt.ylim([0, 5])
            plt.xlabel('Horizontal Location (mm)')
            plt.ylabel('EPL (mm)')
            plt.title('{0} mm Downstream - {1} & {2}'
                      .format(yp[n], test, scan[n]))
            plt.legend()
            plt.savefig('{0}/comparison_{1}mm.png'.format(mb_fld, yp[n]))
            plt.close()

        plt.figure()
        plt.imshow(img, vmin=0, vmax=5)
        plt.colorbar()
        plt.title('EPL [mm] Mapping of Spray - {0}'.format(test))
        for n, _ in enumerate(indices):
            plt.plot(
                     np.linspace(1,768,768),
                     np.linspace(wbp[n], wbp[n], 768),
                     label='{0}'.format(scan[n])
                     )
        plt.legend()
        plt.savefig('{0}/Spray Image.png'.format(mb_fld))
        plt.close()

        with open('{0}/monobeam_data.pckl'.format(mb_fld), 'wb') as f:
            pickle.dump([EPL, scan, wbp, yp], f)


# Run this script
if __name__ == '__main__':
    main()
