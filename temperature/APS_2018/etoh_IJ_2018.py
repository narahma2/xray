"""
Processes the APS 2018-1 ethanol impinging jet data sets.
Created on Fri Jan 24 15:29:06 2020

@author: rahmann
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter
from general.xray_factor import ItoS
from temperature.temperature_processing import main as temperature_processing


def main():
    # Setup
    prj_fld = '/mnt/r/X-ray Temperature/APS 2018-1/'

    test = 'Ethanol/IJ65C'

    folder = prj_fld + 'Processed/' + test
    if not os.path.exists(folder):
        os.makedirs(folder)

    scans = ['/Perpendicular', '/Transverse']

    # Load calibration data set
    with open(glob.glob(prj_fld + '/Processed/Ethanol_700umNozzle/Combined/*.pckl')[0], 'rb') as f:
        temperature_cal, reduced_q_cal, reduced_intensity_cal = pickle.load(f)

    # Select the 65 C temperature case from the calibration set
    cal_intensity_338 = reduced_intensity_cal[np.argmin(abs(temperature_cal - 338))]

    for scan in scans:
        # Create folders
        if not os.path.exists(folder + scan):
            os.makedirs(folder + scan)

        # Load background
        bg = glob.glob(prj_fld + '/Q Space/Ethanol_ImpingingJet/*Scan1152*')
        q = np.loadtxt(bg[0], usecols=0)                    # Load q
        bg = [np.loadtxt(x, usecols=1) for x in bg]         # Load intensity

        # Average background intensities
        bg = np.mean(bg, axis=0)

        # Intensity correction
        perpendicular = [1138, 1139, 1140, 1141, 1142, 1143]
        transverse = [1145, 1146, 1148, 1149, 1150, 1151]

        if 'Perpendicular' in scan:
            files = perpendicular
            y_loc = [5, 8, 11, 14, 17, 20]
        elif 'Transverse' in scan:
            files = transverse
            y_loc = [2, 5, 8, 11, 14, 17]

        files = [glob.glob(prj_fld + '/Q Space/Ethanol_ImpingingJet/*' + str(x) + '*') for x in files]

        intensity = [np.mean([np.loadtxt(x, usecols=1)-bg for x in y], axis=0) for y in files]

        filtered_intensity = [savgol_filter(x, 49, 3) for x in intensity]

        sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())

        reduced_q = q[sl]
        reduced_intensity = [x[sl] for x in filtered_intensity]
        #reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]

        temperature_processing('Ethanol', prj_fld + '/Processed/Ethanol/IJ65C/', scan, reduced_intensity, reduced_q, temperature=None, structure_factor=None, y=y_loc, ramping=False)

        plt.figure()
        [plt.plot(reduced_q, x, color='C'+str(i), label='y = ' + str(y_loc[i]) + ' mm') for i,x in enumerate(reduced_intensity)]
        plt.plot(reduced_q_cal, cal_intensity_338, color='k', linestyle='--', label='Calib. Jet')
        plt.legend()
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
    #    plt.ylim([0.1, 1.1])
        plt.title(scan[1:])
        plt.tight_layout()
        plt.savefig(folder + scan + '/superimposed.png')

        if 'Perpendicular' in scan:
            intensity_perp = reduced_intensity
        elif 'Transverse' in scan:
            intensity_tran = reduced_intensity

    fig = plt.figure(figsize=(9.72, 4.78))
    for i,_ in enumerate(intensity_tran):
        plt.subplot(2,3,i+1)
        plt.plot(reduced_q, intensity_perp[i], color='k', label='Perpendicular', linestyle='--')
        plt.plot(reduced_q, intensity_tran[i], color='k', label='Transverse', linestyle='-')
        plt.plot(reduced_q_cal, cal_intensity_338, color='r', label='Calib. Jet')
        plt.title('y = ' + str(y_loc[i]) + ' mm', fontdict = {'fontsize' : 18})
        plt.xlim([0.6, 1.75])
        plt.ylim([0.2, 2])
        if i == 4:
            plt.xlabel('q (Å$^{-1}$)')
        plt.tight_layout()
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', prop={'size': 'small'})
    fig.suptitle('Ethanol Impinging Jet @ 65 °C', fontsize=18, weight='bold')
    fig.subplots_adjust(top=0.82)
    fig.savefig(folder + '/combined.png')


if __name__ == '__main__':
    main()
