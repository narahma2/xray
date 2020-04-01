# -*- coding: utf-8 -*-
"""
Functions for processing temperature data sets (calibration + IJ).

Created on Tue Jan 21 13:54:00 2020

@author: rahmann
"""

import sys
if sys.platform == 'win32':
    gh_fld = 'E:/GitHub/xray/general'
    sys.path.append(gh_fld)
    sys.path.append('E:/GitHub/xray/temperature')
elif sys.platform == 'linux':
    gh_fld = '/mnt/e/GitHub/xray/general'
    sys.path.append(gh_fld)
    sys.path.append('/mnt/e/GitHub/xray/temperature')

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
fm._rebuild()
import matplotlib.pyplot as plt
plt.style.use(gh_fld + '/python/matplotlib/stylelib/paper.mplstyle')
from PIL import Image
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import stats
from Statistics.calc_statistics import polyfit

# Functions for intensity profile fitting
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def poly4(x, c0, c1, c2, c3, c4):
    return c0*x**4 + c1*x**3 + c2*x**2 + c3*x + c4

def profile(name, fit_var, profile, profiles_folder, stats_folder, test, plots_folder):
    """
    Creates plots for the potential thermometer/positioning profiles.
    =============
    --VARIABLES--
    name:               Profile name ('peak', 'q1', etc.)
    fit_var:            Y axis variable (temperature or y location)
    profile:            X axis variable ('peak', 'q1', etc.)
    profiles_folder:    Location where profiles (X axis values) will be saved.
    stats_folder:       Location where statistics (polynomial fit) will be saved.
    test:               Type of liquid ("Water", "Ethanol", "Dodecane")
    plots_folder        Location where plots will be saved.
    """

    profile_polyfit = polyfit(profile, fit_var, 1)
    np.savetxt(profiles_folder + '/profile_' + name + '.txt', profile)
    np.savetxt(stats_folder + '/' + name + '_polynomial.txt', profile_polyfit['polynomial'])
    
    plt.figure()
    plt.plot(profile, fit_var, ' o', markerfacecolor='none', markeredgecolor='b', label='Data')
    plt.plot(profile, profile_polyfit['function'](profile), 'k', linewidth=2.0, label='y = ' + '%0.2f'%profile_polyfit['polynomial'][0] + 'x + ' + '%0.2f'%profile_polyfit['polynomial'][1])
    plt.title(test + ' ' + name + ' - R$^2$ = ' + str(round(profile_polyfit['determination'],4)))
    plt.legend()
    if any(x in profiles_folder for x in ['IJ Ramping/Temperature', 'IJ Ambient', 'IJ65', 'Cold']):
        plt.ylabel('Y Location (mm)')
    else:
        plt.ylabel('Temperature (K)')
    plt.xlabel(name)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()
    plt.tick_params(which='both',direction='in')
    plt.tight_layout()
    plt.savefig(plots_folder + name + '.png')
    plt.close()
    
def density(test, q_spacing, T):
    """
    Computes density based on q spacing data. Serves as a measure of error.
    =============
    --VARIABLES--
    test:               Type of liquid: "Water", "Ethanol", "Dodecane"
    peak_q:             Location of peak in q [1/A]
    T:                  Temperature of liquid [K]
    """
    
    # Convert q to d (q = 4*sin(theta)*pi/lambda; 2*d*sin(theta) = n*lambda)
    d = 2*pi/peak_q

    # Get temperatures and densities from EES data file
    density_file = sys_folder + '/X-ray Temperature/ScriptData/densities.txt'
    density_data = pd.read_csv(density_file, sep="\t")
    temp = density_data["T_" + test[0:3].lower()]
    dens = density_data["rho" + test[0:3].lower()]

def saveimage(images_folder, fit_var, scatter, background):
    """
    Saves the scatter (w/o subtraction) and background images as 16 bit TIF.
    =============
    --VARIABLES--
    images_folder:       Save location of the TIF files.
    fit_var:            Temperature or y location variable.
    scatter:            Array containing scatter images.
    background:         Array containing background images.
    """

    # Average the background array
    background = np.mean(background, axis=0)

    # Save background
    Image.fromarray(background).save(images_folder + '/BG.tif')

    # Save images
    [Image.fromarray(x).save(images_folder + '/' + '%03i'%n + '_' + ('%06.2f'%fit_var[n]).replace('.', 'p') + '.tif') for n, x in enumerate(scatter)]

    
def main(test, folder, scan, reduced_intensity, reduced_q, temperature=None, structure_factor=None, y=None, ramping=False, scatter=None, background=None):
    """
    Processes data sets, create statistical fits, and outputs plots.
    =============
    --VARIABLES--
    test:               Type of liquid: "Water", "Ethanol", "Dodecane"
    folder:             Save location of the processed data set.
    scan:               Specific scan under the type of test.
    reduced_intensity   Intensity profiles reduced down to the cropped q.
    reduced_q           Cropped q range.
    temperature         Nozzle temperature.
    structure_factor    Structure factor profiles (water only).
    y                   Vertical location in spray (IJ only).
    ramping             Ramping IJ case (True/False).
    """

    profiles_folder = folder + '/' + str(scan) + '/Profiles/'
    if not os.path.exists(profiles_folder):
        os.makedirs(profiles_folder)
        
    stats_folder = folder + '/' + str(scan) + '/Statistics/'
    if not os.path.exists(stats_folder):
        os.makedirs(stats_folder)
        
    plots_folder = folder + '/' + str(scan) + '/Plots/'
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    
    tests_folder = folder + '/' + str(scan) + '/Tests/'
    if not os.path.exists(tests_folder):
        os.makedirs(tests_folder)
        
    curves_folder = folder + '/' + str(scan) + '/Curves/'
    if not os.path.exists(curves_folder):
        os.makedirs(curves_folder)
    
    if structure_factor:
        structure_factor_folder = folder + '/' + str(scan) + '/Structure Factor/'
        if not os.path.exists(structure_factor_folder):
            os.makedirs(structure_factor_folder)
    
    #reduced_intensity = np.array([x / np.max(x) for x in reduced_intensity])

    if 'IJ' in test and 'Ethanol' in test:
        pinned_pts = np.abs(reduced_q - 1.40).argmin()
    else:
        ## Find pinned points in the curves (least variation)
        intensity_std = np.std(reduced_intensity, axis=0)
        pinned_pts = find_peaks(-intensity_std)[0]
        # Find the minimum peak only (throw away every other valley)
        pinned_pts = pinned_pts[np.argmin(intensity_std[pinned_pts])]
        
    pinned_q = reduced_q[pinned_pts] #[reduced_q[int(x)] for x in pinned_pts]

    # Designate fit_var
    if y is not None:
        fit_var = y
        np.savetxt(profiles_folder + '/positions.txt', y)
        np.savetxt(stats_folder + '/positions.txt', y)
    elif temperature is not None:
        fit_var = temperature
        np.savetxt(profiles_folder + '/temperature.txt', temperature)
        np.savetxt(stats_folder + '/temperature.txt', temperature)

    # Save images if scatter and background arrays are passed
    if scatter is not None:
        images_folder = folder + '/' + str(scan) + '/Images/'
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        saveimage(images_folder, fit_var, scatter, background)
    
    profile('peak', fit_var, [np.max(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('peakq', fit_var, [reduced_q[np.argmax(x)] for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('aratio', fit_var, [np.trapz(x[:pinned_pts], reduced_q[:pinned_pts]) / np.trapz(x[pinned_pts:], reduced_q[pinned_pts:]) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('mean', fit_var, [np.mean(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('var', fit_var, [np.var(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('skew', fit_var, [stats.skew(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('kurt', fit_var, [stats.kurtosis(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    
    profile_peakq = np.loadtxt(profiles_folder + '/profile_peakq.txt')

    rr = np.array([(x-min(fit_var))/(max(fit_var)-min(fit_var)) for x in fit_var])
    bb = np.array([1-(x-min(fit_var))/(max(fit_var)-min(fit_var)) for x in fit_var])

    if 'Water' in test:
        llim = 0.46
    elif 'Ethanol' in test:
        llim = 0.30
    elif 'Dodecane' in test:
        llim = 0.10
    else:
        llim = 0.46

    for i,_ in enumerate(reduced_intensity):
        # if temperature is not None:
        #     np.savetxt(tests_folder + '/' + f"{i:02d}" + '_' + str(round(fit_var[i],2)).replace('.','p') + 'C' + '.txt', reduced_intensity[i])
        # elif y is not None:
        #     np.savetxt(tests_folder + '/' + f"{i:02d}" + '_' + str(round(fit_var[i],2)).replace('.','p') + 'mm' + '.txt', reduced_intensity[i])
        
        # Create intensity plots with q values of interest highlighted
        plt.figure()
        plt.plot(reduced_q, reduced_intensity[i], linestyle='-', color=(rr[i],0,bb[i]), linewidth=2.0, label=str(round(fit_var[i],1)) + ' K')
        plt.axvline(x=profile_peakq[i], linestyle='--', color='C1', label='peakq = ' + str(round(profile_peakq[i],2)))
        plt.legend(loc='upper right')
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.autoscale(enable=True, axis='x', tight=True)
        #plt.gca().set_ylim([llim, 1.02])
        plt.minorticks_on()
        plt.tick_params(which='both',direction='in')
        plt.title(test + ' Curves')
        plt.tight_layout()
        plt.savefig(curves_folder +  '/curves_' + str(round(fit_var[i],1)) + '.png')
        plt.close()
            
    
    # if IJ is False:
    if structure_factor:    
        for i,_ in enumerate(structure_factor):
            np.savetxt(tests_folder + '/' + f"{i:02d}" + '_' + str(round(temperature[i],2)).replace('.','p') + 'K' + '.txt', structure_factor[i])
            
            plt.figure()
            plt.plot(reduced_q, structure_factor[i], linestyle='-', color=(rr[i],0,bb[i]), linewidth=2.0, label=str(round(temperature[i],1)) + ' K')
            plt.legend(loc='upper right')
            plt.xlabel('q (Å$^{-1}$)')
            plt.ylabel('Structure Factor (a.u.)')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.minorticks_on()
            plt.tick_params(which='both',direction='in')
            plt.title(test + ' Curves')
            plt.tight_layout()
            plt.savefig(structure_factor_folder +  '/' + test + 'curves_' + str(round(temperature[i],1)) + '.png')
            plt.close()
            
    np.savetxt(folder + '/' + str(scan) + '/q_range.txt', reduced_q)
    if temperature is not None:
        np.savetxt(folder + '/' + str(scan) + '/temperature.txt', temperature)
    elif y is not None:
        np.savetxt(folder + '/' + str(scan) + '/positions.txt', y)
    #%%
    
    if not ramping:
        # Standard deviation plot of all intensities
        plt.figure()
        plt.plot(reduced_q, intensity_std, linewidth='2.0')
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('SD(Intensity) (a.u.)')
        plt.axvline(x=pinned_q, color='k', linestyle='--')
        plt.text(pinned_q, 0.6*np.mean(intensity_std), 'q = ' + "%02.2f"%round(pinned_q, 2), horizontalalignment='center', bbox=dict(facecolor='white', alpha=1.0))
        #[plt.axvline(x=y, color='g', linestyle='-.') for y in peak_q]
        #[plt.text(x, 1.2*np.mean(intensity_std), 'q = ' + "%02.2f"%round(x, 2), horizontalalignment='center', bbox=dict(ec='g',facecolor='white', alpha=1.0)) for x in peak_q]
        plt.title('Scan ' + str(scan))
        plt.tight_layout()
        plt.savefig(plots_folder + 'stdev.png')
        plt.close()
    
    # Superimposed intensity plot    
    plt.figure()
    [plt.plot(reduced_q, x, color=(rr[i],0,bb[i])) for i,x in enumerate(reduced_intensity)]
    plt.xlabel('q (Å$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.axvline(x=pinned_q, color='k', linestyle='--')
    plt.text(pinned_q, 0.5, 'q = ' + "%02.2f"%round(pinned_q, 2), horizontalalignment='center', bbox=dict(facecolor='white', alpha=1.0))
    plt.title('Scan ' + str(scan))
    #plt.ylim([llim, 1.05])
    plt.tight_layout()
    plt.savefig(plots_folder + 'superimposedcurves.png')
    plt.close()
    
    # Save the calibration data sets
    if temperature is not None and ramping is False:
        with open(folder + '/' + str(scan) + '/' + str(scan) + '_data.pckl', 'wb') as f:
            pickle.dump([temperature, reduced_q, reduced_intensity], f)
    # Save the ethanol (cold/ambient/hot) and water impinging jet data sets
    elif y is not None and ramping is False:
        with open(folder + '/' + str(scan) + '/' + str(scan) + '_data.pckl', 'wb') as f:
            pickle.dump([y, reduced_q, reduced_intensity], f)
    # Save the ethanol ramping impinging jet data set
    elif ramping is True:
        with open(folder + '/' + str(scan) + '/' + str(scan).rsplit('/')[-1] + '_data.pckl', 'wb') as f:
            pickle.dump([temperature, y, reduced_q, reduced_intensity], f)
