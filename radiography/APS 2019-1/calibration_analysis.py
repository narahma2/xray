"""
-*- coding: utf-8 -*-
Summarize the correction factors for the LCSC injector.

@Author: rahmann
@Date:   2020-04-30 10:54:07
@Last Modified by:   rahmann
@Last Modified time: 2020-04-30 10:54:07
"""

import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy.signal import savgol_filter, find_peaks, peak_widths
from general.Statistics.calc_statistics import polyfit
from general.misc import create_folder

def get_xpos(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    xpos = np.nanmedian(processed_data['Lateral Position'])

    return xpos


def get_ypos(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    ypos = processed_data['Axial Position']

    return ypos


def get_mean_ellipseT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    mean_ellipseT = np.nanmean(processed_data['Transmission Ratios'][0])

    return mean_ellipseT


def get_ellipseT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    ellipseT = np.array(processed_data['Transmission Ratios'][0])

    return ellipseT


def get_peakT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    peakT = np.array(processed_data['Transmission Ratios'][1])

    return peakT


def get_mean_peakT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    mean_peakT = np.nanmean(processed_data['Transmission Ratios'][1])

    return mean_peakT


def get_peakT(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    peakT = np.array(processed_data['Transmission Ratios'][1])

    return peakT


def get_iface(path):
    with open(path, 'rb') as f:
        processed_data = pickle.load(f)

    iface = np.array(processed_data['Injector Face'])

    return iface


def main():
    # Coding logic:
    #   X Read in all file names
    #   O Read in KI parameters:
    #       O Read ratio_peakT
    #       O Read ratio_ellipseT

    # Location of APS 2019-1 data
    prj_fld = '/mnt/r/X-ray Radiography/APS 2019-1/'

    # Save location for the plots
    plt_fld = create_folder('{0}/Figures/Cal_Summary/'.format(prj_fld))

    # Test matrix
    test_matrix = pd.read_csv('{0}/test_matrix.txt'.format(prj_fld),
                              sep='\t+', engine='python')

    # Processed data sets location
    proc_fld = '{0}/Processed/YAG/Summary/'.format(prj_fld)

    # Read in all file names
    water_tests = glob.glob(proc_fld + '/*Calibration2*')
    ki_tests = glob.glob(proc_fld + '/*KI*')

    # Read in axial positions
    water_apos = [get_ypos(x) for x in water_tests]
    ki_apos = get_ypos(ki_tests[0])

    # Read in water lateral positions
    posn = np.array([get_xpos(x) for x in water_tests])
    posn[np.abs(250-posn) <= 10] = int(0)
    posn[np.abs(390-posn) <= 10] = int(1)
    posn[np.abs(510-posn) <= 10] = int(2)
    posn = posn.astype(int)

    # Indices corresponding to each of the positions
    ind_left = np.where(posn == 0)[0]
    ind_midl = np.where(posn == 1)[0]
    ind_rght = np.where(posn == 2)[0]

    # Indices of only the top injector face (iface = 16)
    top_ind = np.concatenate((ind_left[-2:], ind_midl[-2:], ind_rght[-2:]))

    # Setup water tags for plots 
    lbls = ['Left', 'Middle', 'Right']
    colr = ['lightcoral', 'forestgreen', 'cornflowerblue']
    mrkr = ['s', 'o', '^']

    # Read in transmission ratios
    water_elpsT = np.array([get_ellipseT(x) for x in water_tests])
    ki_elpsT = np.array(get_ellipseT(ki_tests[0]))
    water_peakT = np.array([get_peakT(x) for x in water_tests])
    ki_peakT = np.array(get_peakT(ki_tests[0]))

    # Summarize the CF
    water_elpsCF = np.array([np.nanmedian(x) for x in water_elpsT])
    water_peakCF = np.array([np.nanmedian(x) for x in water_peakT])
    ki_elpsCF = np.nanmedian(ki_elpsT)
    ki_peakCF = np.nanmedian(ki_peakT)

    # Consolidate water CF into only the top injector face (iface = 16)
    water_elpsCF_top = np.median(water_elpsCF[top_ind])
    water_peakCF_top = np.median(water_peakCF[top_ind])

    # Read in Injector Face locations
    iface = np.array([get_iface(x) for x in water_tests])

    # Save the CF
    np.savetxt(
               '{0}/cf_water.txt'.format(proc_fld),
               np.c_[iface, water_elpsCF, water_peakCF],
               delimiter='\t',
               header='IFace\telpsCF\tpeakCF'
               )

    np.savetxt(
               '{0}/cf_summary.txt'.format(proc_fld),
               (water_elpsCF_top, water_peakCF_top, ki_elpsCF, ki_peakCF),
               delimiter='\t',
               header='Water elpsCF\tWater peakCF\tKI elpsCF\tKI peakCF'
               )

    ##########################################################################
    ## PLOTS (O - Planned, X - Completed)
    ##      X--> Vertical EllipseT (combined)
    ##              X--> FIG1: Water
    ##              X--> FIG2: KI
    ##      X--> Vertical EllipseT (left/middle/right subplots)
    ##              X--> FIG3: Water
    ##      X--> Vertical PeakT (combined)
    ##              X--> FIG4: Water
    ##              X--> FIG5: KI
    ##      X--> Vertical PeakT (left/middle/right subplots)
    ##              X--> FIG6: Water
    ##########################################################################

    # FIG1: Water vertical EllipseT
    plt.figure()
    [
     plt.plot(
              water_apos[i],
              water_elpsT[i],
              marker=mrkr[posn[i]],
              markerfacecolor=colr[posn[i]],
              markeredgecolor=colr[posn[i]],
              linestyle='',
              )
     for i,_ in enumerate(water_tests)
     ]
    leg_el = [
              Line2D([0], [0], color=colr[0], marker=mrkr[0], linestyle='',
                     markeredgecolor=colr[0], label=lbls[0]),
              Line2D([0], [0], color=colr[1], marker=mrkr[1], linestyle='',
                     markeredgecolor=colr[1], label=lbls[1]),
              Line2D([0], [0], color=colr[2], marker=mrkr[2], linestyle='',
                     markeredgecolor=colr[2], label=lbls[2])
              ]
    plt.legend(handles=leg_el)
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Correction Factor (-)')
    plt.title('Water EllipseT')
    plt.savefig('{0}/water_vert_elpsT.png'.format(plt_fld))

    # FIG2: KI vertical EllipseT 
    plt.figure()
    plt.plot(
             ki_apos,
             ki_elpsT,
             marker='o',
             linestyle=''
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Correction Factor (-)')
    plt.title('KI EllipseT')
    plt.savefig('{0}/ki_vert_elpsT.png'.format(plt_fld))
    plt.close()

    # FIG3: Vertical EllipseT (left/middle/right)
    plt.subplots(3, 1, sharex=True, figsize=(15,15))
    # Water vertical EllipseT (left)
    plt.subplot(311)
    [
     plt.plot(
              water_apos[i],
              water_elpsT[i],
              marker=mrkr[posn[i]],
              linestyle='',
              label='{0}'.format(iface[i])
              )
     for i in ind_left
     ]
    plt.legend()
    plt.ylabel('Correction Factor (-)')
    plt.title('Water EllipseT Left')
    # Water vertical EllipseT (middle)
    plt.subplot(312)
    [
     plt.plot(
              water_apos[i],
              water_elpsT[i],
              marker=mrkr[posn[i]],
              linestyle='',
              label='{0}'.format(iface[i])
              )
     for i in ind_midl
     ]
    plt.legend()
    plt.ylabel('Correction Factor (-)')
    plt.title('Water EllipseT Middle')
    # Water vertical EllipseT (right)
    plt.subplot(313)
    [
     plt.plot(
              water_apos[i],
              water_elpsT[i],
              marker=mrkr[posn[i]],
              linestyle='',
              label='{0}'.format(iface[i])
              )
     for i in ind_rght
     ]
    plt.legend()
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Correction Factor (-)')
    plt.title('Water EllipseT Right')
    plt.savefig('{0}/water_vert_separate_elpsT.png'.format(plt_fld))

    # FIG4: Water vertical PeakT
    plt.figure()
    [
     plt.plot(
              water_apos[i],
              water_peakT[i],
              marker=mrkr[posn[i]],
              markerfacecolor=colr[posn[i]],
              markeredgecolor=colr[posn[i]],
              linestyle='',
              )
     for i,_ in enumerate(water_tests)
     ]
    leg_el = [
              Line2D([0], [0], color=colr[0], marker=mrkr[0], linestyle='',
                     markeredgecolor=colr[0], label=lbls[0]),
              Line2D([0], [0], color=colr[1], marker=mrkr[1], linestyle='',
                     markeredgecolor=colr[1], label=lbls[1]),
              Line2D([0], [0], color=colr[2], marker=mrkr[2], linestyle='',
                     markeredgecolor=colr[2], label=lbls[2])
              ]
    plt.legend(handles=leg_el)
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Correction Factor (-)')
    plt.title('Water PeakT')
    plt.savefig('{0}/water_vert_peakT.png'.format(plt_fld))

    # FIG5: KI vertical EllipseT 
    plt.figure()
    plt.plot(
             ki_apos,
             ki_peakT,
             marker='o',
             linestyle=''
             )
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Correction Factor (-)')
    plt.title('KI PeakT')
    plt.savefig('{0}/ki_vert_peakT.png'.format(plt_fld))
    plt.close()

    # FIG6: Vertical PeakT (left/middle/right)
    plt.subplots(3, 1, sharex=True, figsize=(15,15))
    # Water vertical PeakT (left)
    plt.subplot(311)
    [
     plt.plot(
              water_apos[i],
              water_peakT[i],
              marker=mrkr[posn[i]],
              linestyle='',
              label='{0}'.format(iface[i])
              )
     for i in ind_left
     ]
    plt.legend()
    plt.ylabel('Correction Factor (-)')
    plt.title('Water PeakT Left')
    # Water vertical PeakT (middle)
    plt.subplot(312)
    [
     plt.plot(
              water_apos[i],
              water_peakT[i],
              marker=mrkr[posn[i]],
              linestyle='',
              label='{0}'.format(iface[i])
              )
     for i in ind_midl
     ]
    plt.legend()
    plt.ylabel('Correction Factor (-)')
    plt.title('Water PeakT Middle')
    # Water vertical PeakT (right)
    plt.subplot(313)
    [
     plt.plot(
              water_apos[i],
              water_peakT[i],
              marker=mrkr[posn[i]],
              linestyle='',
              label='{0}'.format(iface[i])
              )
     for i in ind_rght
     ]
    plt.legend()
    plt.xlabel('Axial Position (px)')
    plt.ylabel('Correction Factor (-)')
    plt.title('Water PeakT Right')
    plt.savefig('{0}/water_vert_separate_peakT.png'.format(plt_fld))


if __name__ == '__main__':
	main()
