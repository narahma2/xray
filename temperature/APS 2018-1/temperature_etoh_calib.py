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
from calc_statistics import polyfit

etoh_folder = 'R:/APS 2018-1/Temperature/Ethanol_700umNozzle/RampUp_qSpace/'
test = 'Ethanol'

folder = 'R:/APS 2018-1/Temperature/Processing/Ethanol_Calibration/'
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
for filename in os.listdir(etoh_folder):
        intensity.append(np.loadtxt(etoh_folder + filename)[:,1])
        q = np.loadtxt(etoh_folder + filename)[:,0]

filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]

temperature_avg = [1, 11, 21, 30, 40, 50, 60, 65]
mid_temp = 4

reduced_q = q[sl]
reduced_intensity = [x[sl] for x in filtered_intensity]
reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
reduced_intensity /= np.max(reduced_intensity)

profile_peak = [np.max(x) for x in reduced_intensity]
profile_peakq = [reduced_q[np.argmax(x)] for x in reduced_intensity]
profile_mean = [np.mean(x) for x in reduced_intensity]
profile_var = [np.var(x) for x in reduced_intensity]
profile_skew = [stats.skew(x) for x in reduced_intensity]
profile_kurt = [stats.kurtosis(x) for x in reduced_intensity]

peak_polyfit = polyfit(profile_peak, temperature_avg, 1)
peakq_polyfit = polyfit(profile_peakq, temperature_avg, 1)
mean_polyfit = polyfit(profile_mean, temperature_avg, 1)
var_polyfit = polyfit(profile_var, temperature_avg, 1)
skew_polyfit = polyfit(profile_skew, temperature_avg, 1)
kurt_polyfit = polyfit(profile_kurt, temperature_avg, 1)

np.savetxt(profiles_folder + '/temperature.txt', temperature_avg)
np.savetxt(profiles_folder + '/profile_peak.txt', profile_peak)
np.savetxt(profiles_folder + '/profile_peakq.txt', profile_peakq)
np.savetxt(profiles_folder + '/profile_mean.txt', profile_mean)
np.savetxt(profiles_folder + '/profile_var.txt', profile_var)
np.savetxt(profiles_folder + '/profile_skew.txt', profile_skew)
np.savetxt(profiles_folder + '/profile_kurt.txt', profile_kurt)

np.savetxt(stats_folder + '/temperature.txt', temperature_avg)
np.savetxt(stats_folder + '/peak_polynomial.txt', peak_polyfit['polynomial'])
np.savetxt(stats_folder + '/peakq_polynomial.txt', peakq_polyfit['polynomial'])
np.savetxt(stats_folder + '/mean_polynomial.txt', mean_polyfit['polynomial'])
np.savetxt(stats_folder + '/var_polynomial.txt', var_polyfit['polynomial'])
np.savetxt(stats_folder + '/skew_polynomial.txt', skew_polyfit['polynomial'])
np.savetxt(stats_folder + '/kurt_polynomial.txt', kurt_polyfit['polynomial'])

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['legend.fontsize'] = 16

plt.figure()
plt.plot(profile_peak, temperature_avg, ' o', markerfacecolor='none', markeredgecolor='b')
plt.plot(profile_peak, peak_polyfit['function'](profile_peak), 'k', linewidth=2.0)
plt.title(test + ' Peak - R$^2$ = ' + str(round(peak_polyfit['determination'],4)))
plt.ylabel('Temperature (°C)')
plt.xlabel('Peak')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'peak.png')

plt.figure()
plt.plot(profile_peakq, temperature_avg, ' o', markerfacecolor='none', markeredgecolor='b')
plt.plot(profile_peakq, peakq_polyfit['function'](profile_peakq), 'k', linewidth=2.0)
plt.title(test + ' Peak Location - R$^2$ = ' + str(round(peakq_polyfit['determination'],4)))
plt.ylabel('Temperature (°C)')
plt.xlabel('Peak Location (Å$^{-1}$)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'peakq.png')

plt.figure()
plt.plot(profile_mean, temperature_avg, ' o', markerfacecolor='none', markeredgecolor='b')
plt.plot(profile_mean, mean_polyfit['function'](profile_mean), 'k', linewidth=2.0)
plt.title(test + ' Mean - R$^2$ = ' + str(round(mean_polyfit['determination'],4)))
plt.ylabel('Temperature (°C)')
plt.xlabel('Mean')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'mean.png')

plt.figure()
plt.plot(profile_var, temperature_avg, ' o', markerfacecolor='none', markeredgecolor='b')
plt.plot(profile_var, var_polyfit['function'](profile_var), 'k', linewidth=2.0)
plt.title(test + ' Variance - R$^2$ = ' + str(round(var_polyfit['determination'],4)))
plt.ylabel('Temperature (°C)')
plt.xlabel('Variance')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'variance.png')

plt.figure()
plt.plot(profile_skew, temperature_avg, ' o', markerfacecolor='none', markeredgecolor='b')
plt.plot(profile_skew, skew_polyfit['function'](profile_skew), 'k', linewidth=2.0)
plt.title(test + ' Skewness - R$^2$ = ' + str(round(skew_polyfit['determination'],4)))
plt.ylabel('Temperature (°C)')
plt.xlabel('Skewness')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'skew.png')

plt.figure()
plt.plot(profile_kurt, temperature_avg, ' o', markerfacecolor='none', markeredgecolor='b')
plt.plot(profile_kurt, kurt_polyfit['function'](profile_kurt), 'k', linewidth=2.0)
plt.title(test + ' Kurtosis - R$^2$ = ' + str(round(kurt_polyfit['determination'],4)))
plt.ylabel('Temperature (°C)')
plt.xlabel('Kurtosis')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.tight_layout()
plt.savefig(plots_folder + 'kurtosis.png')

plt.figure()
plt.plot(reduced_q, reduced_intensity[0], linestyle='-', color='tab:orange', linewidth=2.0, label=str(int(round(temperature_avg[0],1))) + '°C')
plt.plot(reduced_q, reduced_intensity[mid_temp], linestyle='-.', color='tab:blue', linewidth=2.0, label=str(int(round(temperature_avg[mid_temp],1))) + '°C')
plt.plot(reduced_q, reduced_intensity[-1], linestyle=':', color='k', linewidth=2.0, label=str(int(round(temperature_avg[-1],1))) + '°C')
plt.legend()
plt.xlabel('q (Å$^{-1}$)')
plt.ylabel('Intensity (arb. units)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.title('Select ' + test + ' Profiles')
plt.tight_layout()
plt.savefig(plots_folder + 'profiles.png')