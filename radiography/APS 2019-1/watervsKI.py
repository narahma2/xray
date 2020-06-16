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
from general.misc import create_folder


def main():
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'
    test_matrix = pd.read_csv('{0}/test_matrix.txt'.format(prj_fld),
                              sep='\t+', engine='python')
    spray_conditions = ['water', 'KI']
    spray_indices = [25, 35]
    save_fld = create_folder('{0}/Water_vs_KI/'.format(prj_fld))

    water_name = test_matrix['Test'][spray_indices[0]]
    ki_name = test_matrix['Test'][spray_indices[1]]

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

    #%% Plots
    xx = np.linspace(1,768,768)
    xx = xx * cm_px * 10       # in mm
    xx = xx - np.mean(xx)

    yy = np.linspace(0,511,512)
    yy = yy * cm_px * 10       # in mm

    # Time averaged plots
    # FIG1: Horizontal Plot @ y/d = 0.75
    plt.figure()
    plt.plot(
             xx[0::5],
             10000*water[188][0::5],
             color='b',
             marker='o',
             fillstyle='none',
             label='Water Averaged'
             )
    plt.plot(
             xx[0::5],
             10000*ki[188][0::5],
             color='g',
             marker='^',
             fillstyle='none',
             label='KI Averaged'
             )
    plt.title('Time Averaged Comparison at y/d = 0.75')
    plt.xlim([-3, 3])
    plt.ylim([-200, 2500])
    plt.xlabel('Horizontal Location (mm)')
    plt.ylabel('Equivalent Path Length ($\mu$m)')
    plt.legend()
    plt.savefig(save_fld + 'timeaveraged_yd075.png')
    plt.close()

    # FIG2: Horizontal Plot @ y/d = 1.25
    plt.figure()
    plt.plot(
             xx[0::5],
             10000*water[314][0::5],
             color='b',
             marker='o',
             fillstyle='none',
             label='Water Averaged'
             )
    plt.plot(
             xx[0::5],
             10000*ki[314][0::5],
             color='g',
             marker='^',
             fillstyle='none',
             label='KI Averaged'
             )
    plt.title('Time Averaged Comparison at y/d = 1.25')
    plt.xlim([-3, 3])
    plt.ylim([-200, 2500])
    plt.xlabel('Horizontal Location (mm)')
    plt.ylabel('Equivalent Path Length ($\mu$m)')
    plt.legend()
    plt.savefig(save_fld + 'timeaveraged_yd125.png')
    plt.close()

    # FIG3: Horizontal Plot @ y/d = 1.75
    plt.figure()
    plt.plot(
             xx[0::5],
             10000*water[439][0::5],
             color='b',
             marker='o',
             fillstyle='none',
             label='Water Averaged'
             )
    plt.plot(
             xx[0::5],
             10000*ki[439][0::5],
             color='g',
             marker='^',
             fillstyle='none',
             label='KI Averaged'
             )
    plt.title('Time Averaged Comparison at y/d = 1.75')
    plt.xlim([-3, 3])
    plt.ylim([-200, 2500])
    plt.xlabel('Horizontal Location (mm)')
    plt.ylabel('Equivalent Path Length ($\mu$m)')
    plt.legend()
    plt.savefig(save_fld + 'timeaveraged_yd175.png')
    plt.close()

    # FIG4: Plot down the centerline 
    plt.figure()
    plt.plot(
             yy[54:-30][0::5],
             10000*water[54:-30,water_mdpt][0::5],
             color='b',
             marker='o',
             fillstyle='none',
             label='Water Averaged'
             )
    plt.plot(
             yy[54:-30][0::5],
             10000*ki[54:-30,ki_mdpt][0::5],
             color='g',
             marker='^',
             fillstyle='none',
             label='KI Averaged'
             )
    plt.title('Time Averaged Comparison Down Centerline')
    plt.xlim([yy[50], yy[-25]])
    plt.ylim([1000, 3500])
    plt.xlabel('Vertical Location (mm)')
    plt.ylabel('Equivalent Path Length ($\mu$m)')
    plt.legend()
    plt.savefig(save_fld + 'timeaveraged_centerline.png')
    plt.close()


# Run this script
if __name__ == '__main__':
    main()
