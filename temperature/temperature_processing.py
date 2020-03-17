# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:54:00 2020

@author: rahmann
"""

import sys
if sys.platform == 'win32':
    sys_folder = 'R:/'
elif sys.platform == 'linux':
    sys_folder = '/mnt/r/'

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy import stats
from calc_statistics import polyfit

# Plot defaults
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['legend.fontsize'] = 16

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
    plt.plot(profile, profile_polyfit['function'](profile), 'k', linewidth=2.0, label='y = ' + '%0.2f'%profile_polyfit['polynomial'][0] + ' + ' + '%0.2f'%profile_polyfit['polynomial'][1])
    plt.title(test + ' ' + name + ' - R$^2$ = ' + str(round(profile_polyfit['determination'],4)))
    plt.legend()
    if 'IJ' in profiles_folder:
        plt.ylabel('Y Location (mm)')
    else:
        plt.ylabel('Temperature (°C)')
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
    T:                  Temperature of liquid [°C]
    """
    
    # Convert q to d (q = 4*sin(theta)*pi/lambda; 2*d*sin(theta) = n*lambda)
    d = 2*pi/peak_q

    # Get temperatures and densities from EES data file
    density_file = sys_folder + '/X-ray Temperature/ScriptData/densities.txt'
    density_data = pd.read_csv(density_file, sep="\t")
    temp = density_data["T_" + test[0:3].lower()]
    dens = density_data["rho" + test[0:3].lower()]
    
def main(test, folder, scan, reduced_intensity, reduced_q, temperature=None, structure_factor=None, y=None, ramping=False):
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
        
    concavity_folder = folder + '/' + str(scan) + '/Concavity/'
    if not os.path.exists(concavity_folder):
        os.makedirs(concavity_folder)
    
    if structure_factor:
        structure_factor_folder = folder + '/' + str(scan) + '/Structure Factor/'
        if not os.path.exists(structure_factor_folder):
            os.makedirs(structure_factor_folder)
    
    reduced_intensity = np.array([x / np.max(x) for x in reduced_intensity])
    
    concavity = [-np.gradient(np.gradient(i)) for i in reduced_intensity]
    concavity = [savgol_filter(j, 55, 3) for j in concavity]
    conc_peak_locs = [find_peaks(k, height=0.00001, distance=100, width=15)[0] for k in concavity]
    conc_q1 = np.array([reduced_q[x[0]] for x in conc_peak_locs])
    conc_q2 = np.array([reduced_q[x[1]] if len(x) == 2 else np.nan for x in conc_peak_locs])   

    #%% Gaussian fitting
    peak_q = [reduced_q[np.argmax(x)] for x in reduced_intensity]
    peak = [np.max(x) for x in reduced_intensity]
    
    # Expected/Bounds: [mu1,sigma1,A1,mu2,sigma2,A2]
    # mu:       location of peak
    # sigma:    width of peak
    # A:        height of peak
    
    if 'Water' in test:
        expected = [(peak_q[i],0.4,peak[i],  2.7,0.3,0.5) for i,_ in enumerate(peak)]
        bounds = [((peak_q[i]-1E-2,-np.inf,peak[i]-1E-2,  2.5,0.2,0.4), (peak_q[i],np.inf,peak[i],  3.3,0.8,0.7)) for i,_ in enumerate(peak)]
    elif 'Ethanol' in test:
        expected = [(0.7,0.3,0.3,  peak_q[i],0.25,peak[i]) for i,_ in enumerate(peak)]
        bounds = [((0.5,0.2,0.3,  peak_q[i]-1E-6,-np.inf,peak[i]-1E-6), (0.9,0.4,0.45,  peak_q[i]+1E-6,np.inf,peak[i]+1E-6)) for i,_ in enumerate(peak)]
    elif 'Dodecane' in test:
        expected = [(peak_q[i],0.3,peak[i]) for i,_ in enumerate(peak)]
        bounds = [((peak_q[i]-1E-6,-np.inf,peak[i]-1E-6), (peak_q[i]+1E-6,np.inf,peak[i]+1E-6)) for i,_ in enumerate(peak)]
    
    if 'Dodecane' not in test:
        fits = [curve_fit(bimodal,reduced_q,x,expected[i],bounds=bounds[i]) for i,x in enumerate(reduced_intensity)]
    elif 'Dodecane' in test:
        fits = [curve_fit(gauss,reduced_q,x,expected[i],bounds=bounds[i]) for i,x in enumerate(reduced_intensity)]
    params = [x[0] for x in fits]
    cov = [x[1] for x in fits]
    sigma = [np.sqrt(np.diag(x)) for x in cov]
    
    mu1 = np.array([x[0] for x in params])
    mu1_int = np.array([reduced_intensity[i][np.abs(x - reduced_q).argmin()] for i,x in enumerate(mu1)])
    sigma1 = np.array([x[1] for x in params])
    A1 = np.array([x[2] for x in params])
    
    if 'Dodecane' not in test:
        mu2 = np.array([x[3] for x in params])
        mu2_int = np.array([reduced_intensity[i][np.abs(x - reduced_q).argmin()] for i,x in enumerate(mu2)])
        sigma2 = np.array([x[4] for x in params])
        A2 = np.array([x[5] for x in params])
    else:
        mu2 = np.zeros(mu1.shape)
        mu2_int = np.zeros(mu1.shape)
        sigma2 = np.zeros(mu1.shape)
        A2 = np.zeros(mu1.shape)
    
    #%% Find pinned points in the curves (least variation)
    intensity_std = np.std(reduced_intensity, axis=0)
    pinned_pts = find_peaks(-intensity_std)[0]
    pinned_q = [reduced_q[int(x)] for x in pinned_pts]
    pinned_q_int1 = [x*0 for x in mu1_int] #np.array([reduced_intensity[i][int(x)] for i,x in enumerate(pinned_pts)][0])
    pinned_q_int2 = [x*0 for x in mu2_int] #np.array([reduced_intensity[i][int(x)] for i,x in enumerate(pinned_pts)][-1])
    peak_pts = find_peaks(intensity_std)[0]
    peak_q = [reduced_q[int(x)] for x in peak_pts]
    
    #%% Polynomial fitting
    poly_sigma = np.ones(len(reduced_q))
#    poly_sigma[pinned_pts[-1]] = 0.01    
    fits = [curve_fit(poly4,reduced_q,x,sigma=poly_sigma) for x in reduced_intensity]
    params = [x[0] for x in fits]
    
    c0 = np.array([x[0] for x in params])
    c1 = np.array([x[1] for x in params])
    c2 = np.array([x[2] for x in params])
    c3 = np.array([x[3] for x in params])
    c4 = np.array([x[4] for x in params])
    
    intensity_poly4 = [c0[i]*reduced_q**4 + c1[i]*reduced_q**3 + c2[i]*reduced_q**2 + c3[i]*reduced_q + c4[i] for i,_ in enumerate(c0)]

    #%%
    if y is not None:
        fit_var = y
        np.savetxt(profiles_folder + '/positions.txt', y)
        np.savetxt(stats_folder + '/positions.txt', y)
    elif temperature is not None:
        fit_var = temperature
        np.savetxt(profiles_folder + '/temperature.txt', temperature)
        np.savetxt(stats_folder + '/temperature.txt', temperature)
    
    profile('peak', fit_var, [np.max(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('peakq', fit_var, [reduced_q[np.argmax(x)] for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('q1', fit_var, mu1, profiles_folder, stats_folder, test, plots_folder)
    profile('q1int', fit_var, mu1_int, profiles_folder, stats_folder, test, plots_folder)
    profile('q2', fit_var, mu2, profiles_folder, stats_folder, test, plots_folder)
    profile('q2int', fit_var, mu2_int, profiles_folder, stats_folder, test, plots_folder)
    profile('qratio', fit_var, mu1 / mu2, profiles_folder, stats_folder, test, plots_folder)
    profile('qdist', fit_var, mu1 - mu2, profiles_folder, stats_folder, test, plots_folder)
    profile('iratio', fit_var, mu1_int / mu2_int, profiles_folder, stats_folder, test, plots_folder)
    #profile('ipratio', fit_var, mu1_int / pinned_q_int2, profiles_folder, stats_folder, test, plots_folder)
    profile('ratio', fit_var, (mu1_int/mu2_int)/(mu1/mu2), profiles_folder, stats_folder, test, plots_folder)
    profile('aratio', fit_var, [np.trapz(x[:pinned_pts[-1]], reduced_q[:pinned_pts[-1]]) / np.trapz(x[pinned_pts[-1]:], reduced_q[pinned_pts[-1]:]) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('mean', fit_var, [np.mean(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('var', fit_var, [np.var(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('skew', fit_var, [stats.skew(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    profile('kurt', fit_var, [stats.kurtosis(x) for x in reduced_intensity], profiles_folder, stats_folder, test, plots_folder)
    
    # Gaussian parameters
    profile('sigma1', fit_var, sigma1, profiles_folder, stats_folder, test, plots_folder)
    profile('sigma2', fit_var, sigma2, profiles_folder, stats_folder, test, plots_folder)
    profile('A1', fit_var, A1, profiles_folder, stats_folder, test, plots_folder)
    profile('A2', fit_var, A2, profiles_folder, stats_folder, test, plots_folder)
    
    # Polynomial parameters
    profile('c0', fit_var, c0, profiles_folder, stats_folder, test, plots_folder)
    profile('c1', fit_var, c1, profiles_folder, stats_folder, test, plots_folder)
    profile('c2', fit_var, c2, profiles_folder, stats_folder, test, plots_folder)
    profile('c3', fit_var, c3, profiles_folder, stats_folder, test, plots_folder)
    profile('c4', fit_var, c4, profiles_folder, stats_folder, test, plots_folder)

    profile_peakq = np.loadtxt(profiles_folder + '/profile_peakq.txt')
    profile_q1 = np.loadtxt(profiles_folder + '/profile_q1.txt')
    profile_q2 = np.loadtxt(profiles_folder + '/profile_q2.txt')

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
        plt.plot(reduced_q, reduced_intensity[i], linestyle='-', color=(rr[i],0,bb[i]), linewidth=2.0, label=str(round(fit_var[i],1)) + '°C')
        plt.plot(reduced_q, intensity_poly4[i], linestyle='-', color='k', linewidth=1.0, label='Poly4')
        plt.axvline(x=profile_peakq[i], linestyle='--', color='C1', label='peakq = ' + str(round(profile_peakq[i],2)))
        plt.axvline(x=profile_q1[i], linestyle='--', color='C2', label='q$_1$ = ' + str(round(profile_q1[i],2)))
        plt.axvline(x=profile_q2[i], linestyle='--', color='C3', label='q$_2$ = ' + str(round(profile_q2[i],2)))
        plt.legend(loc='upper right')
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.gca().set_ylim([llim, 1.02])
        plt.minorticks_on()
        plt.tick_params(which='both',direction='in')
        plt.title(test + ' Curves')
        plt.tight_layout()
        plt.savefig(curves_folder +  '/curves_' + str(round(fit_var[i],1)) + '.png')
        plt.close()
            
    
    # if IJ is False:
    for i,_ in enumerate(concavity):
        if temperature is not None:
            np.savetxt(tests_folder + '/concavity' + f"{i:02d}" + '_' + str(round(fit_var[i],2)).replace('.','p') + 'C.txt', concavity[i])
        elif y is not None:
            np.savetxt(tests_folder + '/concavity' + f"{i:02d}" + '_' + str(round(fit_var[i],2)).replace('.','p') + 'mm.txt', concavity[i])
        
        plt.figure()
        if temperature is not None:
            plt.plot(reduced_q, concavity[i], linestyle='-', color=(rr[i],0,bb[i]), linewidth=2.0, label=str(round(fit_var[i],1)) + '°C')
        elif y is not None:
            plt.plot(reduced_q, concavity[i], linestyle='-', color=(rr[i],0,bb[i]), linewidth=2.0, label=str(round(fit_var[i],1)) + 'mm')
        plt.axvline(x=conc_q1[i], linestyle='--', color='C2', label='q$_1$ = ' + str(round(conc_q1[i],2)))
        plt.axvline(x=conc_q2[i], linestyle='--', color='C3', label='q$_2$ = ' + str(round(conc_q2[i],2)))
        plt.legend(loc='upper right')
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('Negative Concavity (a.u.)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.minorticks_on()
        plt.tick_params(which='both',direction='in')
        plt.title(test + ' Curves')
        plt.tight_layout()
        plt.savefig(concavity_folder +  '/' + test + 'curves_' + str(round(fit_var[i],1)) + '.png')
        plt.close()
        
    if structure_factor:    
        for i,_ in enumerate(structure_factor):
            np.savetxt(tests_folder + '/' + f"{i:02d}" + '_' + str(round(temperature[i],2)).replace('.','p') + 'C' + '.txt', structure_factor[i])
            
            plt.figure()
            plt.plot(reduced_q, structure_factor[i], linestyle='-', color=(rr[i],0,bb[i]), linewidth=2.0, label=str(round(temperature[i],1)) + '°C')
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

    # Standard deviation plot of all intensities
    plt.figure()
    plt.plot(reduced_q, intensity_std, linewidth='2.0')
    plt.xlabel('q (Å$^{-1}$)')
    plt.ylabel('SD(Intensity) [a.u.]')
    [plt.axvline(x=y, color='k', linestyle='--') for y in pinned_q]
    [plt.text(x, 0.6*np.mean(intensity_std), 'q = ' + "%02.2f"%round(x, 2), horizontalalignment='center', bbox=dict(facecolor='white', alpha=1.0)) for x in pinned_q]
    [plt.axvline(x=y, color='g', linestyle='-.') for y in peak_q]
    [plt.text(x, 1.2*np.mean(intensity_std), 'q = ' + "%02.2f"%round(x, 2), horizontalalignment='center', bbox=dict(ec='g',facecolor='white', alpha=1.0)) for x in peak_q]
    plt.title('Scan ' + str(scan))
    plt.tight_layout()
    plt.savefig(plots_folder + 'stdev.png')
    #    plt.close()
    
    # Superimposed intensity plot    
    plt.figure()
    [plt.plot(reduced_q, x, color=(rr[i],0,bb[i])) for i,x in enumerate(reduced_intensity)]
    plt.xlabel('q (Å$^{-1}$)')
    plt.ylabel('Intensity [a.u.]')
    [plt.axvline(x=y, color='k', linestyle='--') for y in pinned_q]
    [plt.text(x, 0.5, 'q = ' + "%02.2f"%round(x, 2), horizontalalignment='center', bbox=dict(facecolor='white', alpha=1.0)) for x in pinned_q]
    [plt.axvline(x=y, color='g', linestyle='-.') for y in peak_q]
    [plt.text(x, 0.7, 'q = ' + "%02.2f"%round(x, 2), horizontalalignment='center', bbox=dict(ec='g',facecolor='white', alpha=1.0)) for x in peak_q]
    plt.title('Scan ' + str(scan))
    plt.ylim([llim, 1.05])
    plt.tight_layout()
    plt.savefig(plots_folder + 'superimposedcurves.png')
    #    plt.close()
    
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
        with open(folder + '/' + str(scan) + '_data.pckl', 'wb') as f:
            pickle.dump([temperature, y, reduced_q, reduced_intensity], f)