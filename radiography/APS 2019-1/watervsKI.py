# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:15:45 2019

@author: rahmann
"""

import os
import h5py
import pickle
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from general.spectrum_modeling import density_KIinH2O
from general.misc import create_folder


def main():
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'
    test_matrix = pd.read_csv('{0}/test_matrix.txt'.format(prj_fld),
                              sep='\t+', engine='python')
    spray_conditions = ['water', 'KI']
    spray_indices = [25, 35]
    save_fld = create_folder('{0}/Water_vs_KI/'.format(prj_fld))
    hdf5_fld = '{0}/HDF5'.format(prj_fld)

    water_name = test_matrix['Test'][spray_indices[0]]
    ki_name = test_matrix['Test'][spray_indices[1]]

    water_test = water_name.rsplit('_')[0]
    ki_test = ki_name.rsplit('_')[0]

    cm_px = 0.05 / 52   # See 'Pixel Size' in Excel workbook

    # Open water & KI spray images
    water = np.array(Image.open('{0}/Images/Spray/EPL/TimeAvg/AVG_{1}.tif'
                                .format(prj_fld, water_name)))
    ki = np.array(Image.open('{0}/Images/Spray/EPL/TimeAvg/AVG_{1}.tif'
                             .format(prj_fld, ki_name)))

    # Find middle of the spray (should be around 768/2=384)
    # Water midpoint
    spray_middle = np.zeros((512,1), dtype=float)
    for i, m in enumerate(water):
        if i >= 35:
            filtered = savgol_filter(m[100:-10], 25, 3)
            spray_middle[i] = np.argmax(filtered) + 99

    water_mdpt = int(round(np.mean(spray_middle[35:])))

    # KI midpoint
    spray_middle = np.zeros((512,1), dtype=float)
    for i, m in enumerate(ki):
        if i >= 35:
            filtered = savgol_filter(m[100:-10], 25, 3)
            spray_middle[i] = np.argmax(filtered) + 99

    ki_mdpt = int(round(np.mean(spray_middle[35:])))

    # Load HDF5 monobeam scan for J ~ 3.26
    scans = [5124, 5125, 5126]
    yp = np.array([1.15, 2.3, 4.6])
    wbp = yp / (cm_px * 10)

    EPL = len(scans) * [None]
    x = len(scans) * [None]

    for n, scan in enumerate(scans):
        f = h5py.File('{0}/Scan_{1}.hdf5'.format(hdf5_fld, scan), 'r')
        x[n] = np.array(f['X'])
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

        EPL[n] = extinction_length / atten_coeff

    #%% Plots
    xx = np.linspace(1,768,768)
    xx = xx * cm_px * 10       # in mm
    xx = xx - np.mean(xx)

    yy = np.linspace(0,511,512)
    yy = yy * cm_px * 10       # in mm

    # Time averaged plots
    # FIG1: Horizontal Plot @ y = 1.15 mm
    plt.figure()
    plt.plot(
             xx[0::5],
             10000*water[125][0::5],
             color='b',
             marker='o',
             fillstyle='none',
             label='Water WB {0}'.format(water_test)
             )
    plt.plot(
             xx[0::5],
             10000*ki[125][0::5],
             color='g',
             marker='^',
             fillstyle='none',
             label='KI WB {0}'.format(ki_test)
             )
    plt.plot(
             x[0],
             1000*np.mean(EPL[0], axis=1),
             color='r',
             marker='s',
             fillstyle='none',
             label='Water MB {0}'.format(scans[0])
             )
    plt.title('Time Averaged Comparison at y = 1.15 mm')
    plt.xlim([-3, 3])
    plt.ylim([-200, 3500])
    plt.xlabel('Horizontal Location (mm)')
    plt.ylabel('Equivalent Path Length ($\mu$m)')
    plt.legend()
    plt.savefig(save_fld + 'timeaveraged_y115.png')
    plt.close()

    # FIG2: Horizontal Plot @ y = 2.30
    plt.figure()
    plt.plot(
             xx[0::5],
             10000*water[251][0::5],
             color='b',
             marker='o',
             fillstyle='none',
             label='Water WB {0}'.format(water_test)
             )
    plt.plot(
             xx[0::5],
             10000*ki[251][0::5],
             color='g',
             marker='^',
             fillstyle='none',
             label='KI WB {0}'.format(ki_test)
             )
    plt.plot(
             x[1],
             1000*np.mean(EPL[1], axis=1),
             color='r',
             marker='s',
             fillstyle='none',
             label='Water MB {0}'.format(scans[1])
             )
    plt.title('Time Averaged Comparison at y = 2.30')
    plt.xlim([-3, 3])
    plt.ylim([-200, 3500])
    plt.xlabel('Horizontal Location (mm)')
    plt.ylabel('Equivalent Path Length ($\mu$m)')
    plt.legend()
    plt.savefig(save_fld + 'timeaveraged_y230.png')
    plt.close()

    # FIG3: Horizontal Plot @ y = 4.60
    plt.figure()
    plt.plot(
             xx[0::5],
             10000*water[502][0::5],
             color='b',
             marker='o',
             fillstyle='none',
             label='Water WB {0}'.format(water_test)
             )
    plt.plot(
             xx[0::5],
             10000*ki[502][0::5],
             color='g',
             marker='^',
             fillstyle='none',
             label='KI WB {0}'.format(ki_test)
             )
    plt.plot(
             x[0],
             1000*np.mean(EPL[2], axis=1),
             color='r',
             marker='s',
             fillstyle='none',
             label='Water MB {0}'.format(scans[2])
             )
    plt.title('Time Averaged Comparison at y = 4.60')
    plt.xlim([-3, 3])
    plt.ylim([-200, 3500])
    plt.xlabel('Horizontal Location (mm)')
    plt.ylabel('Equivalent Path Length ($\mu$m)')
    plt.legend()
    plt.savefig(save_fld + 'timeaveraged_y460.png')
    plt.close()

    # FIG4: Plot down the centerline
    plt.figure()
    plt.plot(
             yy[54:-30][0::5],
             10000*water[54:-30, water_mdpt][0::5],
             color='b',
             marker='o',
             fillstyle='none',
             label='Water {0} @ {1}'.format(water_test, water_mdpt)
             )
    plt.plot(
             yy[54:-30][0::5],
             10000*ki[54:-30, ki_mdpt][0::5],
             color='g',
             marker='^',
             fillstyle='none',
             label='KI {0} @ {1}'.format(ki_test, ki_mdpt)
             )
    plt.title('Time Averaged Comparison Down Centerline')
    plt.xlim([yy[50], yy[-25]])
    plt.ylim([-200, 3500])
    plt.xlabel('Vertical Location (mm)')
    plt.ylabel('Equivalent Path Length ($\mu$m)')
    plt.legend()
    plt.savefig(save_fld + 'timeaveraged_centerline.png')
    plt.close()


# Run this script
if __name__ == '__main__':
    main()
