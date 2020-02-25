# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:13:08 2019

@author: rahmann
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats

ij_folder = 'R:/APS 2018-1/Temperature/Ethanol_ImpingingJet/q Space/'

folder = 'R:/APS 2018-1/Temperature/Processing/Ethanol_ImpingingJet/'
if not os.path.exists(folder):
    os.makedirs(folder)

profiles_folder = folder + '/Profiles/'
if not os.path.exists(profiles_folder):
    os.makedirs(profiles_folder)
    
stats_folder = folder + '/Statistics/'
if not os.path.exists(stats_folder):
    os.makedirs(stats_folder)
    
plots_folder = folder + '/Plots/'
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

sl = slice(108, 432)

intensity = []
for filename in os.listdir(ij_folder):
        intensity.append(np.loadtxt(ij_folder + filename)[:,1])
        q = np.loadtxt(ij_folder + filename)[:,0]

filtered_intensity_perp = [savgol_filter(x, 55, 3) for x in intensity[0:7]]
filtered_intensity_trans = [savgol_filter(x, 55, 3) for x in intensity[7:]]

location_perp = [2, 5, 8, 11, 14, 17, 20]
location_trans = [2, 5, 8, 11, 14, 17]

location = location_trans

reduced_q = q[sl]
reduced_intensity = [x[sl] for x in filtered_intensity_trans]
reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
reduced_intensity /= np.max(reduced_intensity)

profile_peak = [np.max(x) for x in reduced_intensity]
profile_peakq = [reduced_q[np.argmax(x)] for x in reduced_intensity]
profile_mean = [np.mean(x) for x in reduced_intensity]
profile_var = [np.var(x) for x in reduced_intensity]
profile_skew = [stats.skew(x) for x in reduced_intensity]
profile_kurt = [stats.kurtosis(x) for x in reduced_intensity]

np.savetxt(profiles_folder + '/profile_peak.txt', profile_peak)
np.savetxt(profiles_folder + '/profile_peakq.txt', profile_peakq)
np.savetxt(profiles_folder + '/profile_mean.txt', profile_mean)
np.savetxt(profiles_folder + '/profile_var.txt', profile_var)
np.savetxt(profiles_folder + '/profile_skew.txt', profile_skew)
np.savetxt(profiles_folder + '/profile_kurt.txt', profile_kurt)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['legend.fontsize'] = 16

plt.figure()
plt.plot(profile_peak, location, ' o', markerfacecolor='none', markeredgecolor='b')
plt.title('Peak')
plt.ylabel('Location (mm)')
plt.xlabel('Peak')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'peak.png')

plt.figure()
plt.plot(profile_peakq, location, ' o', markerfacecolor='none', markeredgecolor='b')
plt.title('Peak Location')
plt.ylabel('Location (mm)')
plt.xlabel('Peak Location (Å$^{-1}$)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'peakq.png')

plt.figure()
plt.plot(profile_mean, location, ' o', markerfacecolor='none', markeredgecolor='b')
plt.title('Mean')
plt.ylabel('Location (mm)')
plt.xlabel('Mean')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'mean.png')

plt.figure()
plt.plot(profile_var, location, ' o', markerfacecolor='none', markeredgecolor='b')
plt.title('Variance')
plt.ylabel('Location (mm)')
plt.xlabel('Variance')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'variance.png')

plt.figure()
plt.plot(profile_skew, location, ' o', markerfacecolor='none', markeredgecolor='b')
plt.title('Skewness')
plt.ylabel('Location (mm)')
plt.xlabel('Skewness')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'skew.png')

plt.figure()
plt.plot(profile_kurt, location, ' o', markerfacecolor='none', markeredgecolor='b')
plt.title('Kurtosis')
plt.ylabel('Location (mm)')
plt.xlabel('Kurtosis')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'kurtosis.png')

plt.figure()
i = -1
for x in reduced_intensity:
    i += 1
    plt.plot(reduced_q, x, label='y = ' + str(location[i]) + ' mm')
plt.xlabel('q (Å$^{-1}$)')
plt.ylabel('Intensity (arb. units)')
plt.title('Ethanol Impinging Jet Profiles')
plt.legend()
plt.tight_layout()
plt.savefig(plots_folder + 'profiles.png')