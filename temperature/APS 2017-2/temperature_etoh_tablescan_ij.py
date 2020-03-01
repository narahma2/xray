# -*- coding: utf-8 -*-
"""
Processes the ethanol impinging jet table scan (425).
See "Ramping Temperature Table Scan 425 for EtOH Impinging Jet" in <Impinging Jet - Ethanol.txt>.

Created on Sat Apr  6 10:15:38 2019

@author: rahmann
"""

import sys
if sys.platform == 'win32':
	sys.path.append('E:/GitHub/xray/general')
	sys_folder = 'R:/'
elif sys.platform == 'linux':
	sys.path.append('/mnt/e/GitHub/xray/general')
	sys_folder = '/mnt/r/'

import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from calc_statistics import comparefit

test = 'Ethanol Impinging Jet Table Scan'
calib_no = [408]
rank_x = np.linspace(0,13, num=14, dtype=int)

project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'

f = h5py.File(project_folder + '/RawData/Scan_' + str(425) + '.hdf5', 'r')
g = h5py.File(project_folder + '/RawData/Scan_' + str(429) + '.hdf5', 'r')
bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
bg_avg = np.mean(bg, axis=0)

q = list(f['q'])
intensities = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]

sl = slice((np.abs(np.array(q) - 0.6)).argmin(), (np.abs(np.array(q) - 1.75)).argmin())

nozzle_T = []
y_loc = []

for i in rank_x:
    y_locs = list(f['Rank_2_Point_' + str(i) + '/7bmb1:aero:m1.VAL'])
    for n, y in enumerate(y_locs):
        nozzle_T.append(list(f['Rank_2_Point_' + str(i) + '/7bm_dau1:dau:010:ADC'])[n])
        y_loc.append(round(list(f['Rank_2_Point_' + str(i) + '/7bmb1:aero:m1.VAL'])[n], 2))
        
positions = []
for k in y_locs:
    positions.append([i for i, x in enumerate(y_loc) if x == round(k, 2)])      # 14 total occurrences of the 18 y positions
        
intensities = intensities[0:252]
intensities = [(x-bg_avg) for x in intensities]
filtered_intensity = [savgol_filter(x, 55, 3) for x in intensities]
reduced_q = q[sl]
reduced_intensity = [x[sl] for x in filtered_intensity]
reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
#reduced_intensity = [z/np.max(z) for z in reduced_intensity]

maxVal = []
for i in range(len(positions)):
    maxVal.append(np.max([reduced_intensity[z] for z in positions[i]]))

for i in range(len(reduced_intensity)):
    find_yposition = (np.abs(y_loc[i] - np.array(y_locs))).argmin()
    reduced_intensity[i] = reduced_intensity[i] / maxVal[find_yposition]

peakq1 = []
peakq2 = []
peakq_ratio = []
peaks_info = []
var = []
skew = []
kurt = []
peak = []
for j in reduced_intensity:
    peakq1.append(reduced_q[find_peaks(j, distance=100, width=20)[0][0]])
    if len(find_peaks(j, distance=100, width=20)[0]) == 1:
        peakq2.append(np.nan)
    if len(find_peaks(j, distance=100, width=20)[0]) == 2:
        peakq2.append(reduced_q[find_peaks(j, distance=100, width=20)[0][1]])
    peakq_ratio.append(peakq2[-1] / peakq1[-1])
    var.append(np.var(j))
    skew.append(stats.skew(j))
    kurt.append(stats.kurtosis(j))
    peak.append(np.max(j))
    
for calib in calib_no:
    calib_folder = project_folder + '/Processed/Ethanol' + str(calib) + '/Statistics'
    calib_var = np.poly1d(np.loadtxt(calib_folder + '/var_polynomial.txt'))
    calib_skew = np.poly1d(np.loadtxt(calib_folder + '/skew_polynomial.txt'))
    calib_kurt = np.poly1d(np.loadtxt(calib_folder + '/kurt_polynomial.txt'))
    calib_peakq2 = np.poly1d(np.loadtxt(calib_folder + '/peakq_polynomial.txt'))
    
    calc_stats = [var, skew, kurt, peakq2]
    stats = [calib_var, calib_skew, calib_kurt, calib_peakq2]
    stats_name = ['var', 'skew', 'kurt', 'peakq2']
    stats_title = ['Variance', 'Skewness', 'Kurtosis', 'Peak Location']
    
    r2_var = []
    r2_skew = []
    r2_kurt = []
    r2_peakq2 = []
    rmse_var = []
    rmse_skew = []
    rmse_kurt = []
    rmse_peakq2 = []
    
    for i in range(len(y_locs)):
        folder = project_folder + '/Processed/Ethanol/IJ Ramping/y' + str(round(y_locs[i], 2))
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        for k, j in enumerate(stats):
            calib_curve = j([calc_stats[k][z] for z in positions[i]])
            calib_curve[calib_curve > 100] = np.nan
            calib_curve[calib_curve < 0] = np.nan
            
            plt.figure()
            plt.plot(np.linspace(1,14,14),calib_curve, ' o', markerfacecolor='none', markeredgecolor='b')
            plt.plot(np.linspace(1,14,14),[nozzle_T[z] for z in positions[i]])
#            plt.title('Temperature from ' + stats_title[k] + ' - y = ' + str(round(y_locs[i],2)) + 'mm')
            plt.ylabel('Temperature (°C)')
            plt.xlabel('Scan')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.minorticks_on()
            if y_locs[i] > 5:
                plt.gca().set_ylim([0, 90])
            else:
                plt.gca().set_ylim([0, 70])
            plt.gca().tick_params(axis='x', which='minor', bottom=False)
            plt.gca().set_xticks(np.linspace(1,14,14))
            plt.tick_params(which='both',direction='in')
            plt.tight_layout()
            plt.savefig(folder + '/' + str(calib) + '_' + stats_name[k] + '.png')
            plt.close()

        # Compare fit for variance
        det, rmse = comparefit(calib_var([var[z] for z in positions[i]]), [nozzle_T[z] for z in positions[i]])
        r2_var.append(det)
        rmse_var.append(rmse)

        # Compare fit for skewness
        det, rmse = comparefit(calib_skew([skew[z] for z in positions[i]]), [nozzle_T[z] for z in positions[i]])
        r2_skew.append(det)
        rmse_skew.append(rmse)
        
        # Compare fit for kurtosis
        det, rmse = comparefit(calib_kurt([kurt[z] for z in positions[i]]), [nozzle_T[z] for z in positions[i]])
        r2_kurt.append(det)
        rmse_kurt.append(rmse)
        
        # Compare fit for peak q
        det, rmse = comparefit(calib_peakq2([peakq2[z] for z in positions[i]]), [nozzle_T[z] for z in positions[i]])
        r2_peakq2.append(det)
        rmse_peakq2.append(rmse)

etoh_q = np.loadtxt(project_folder + '/Processed/Ethanol/' + str(calib) + '/q_range.txt')
etoh_calib_low = np.loadtxt(project_folder + '/Processed/Ethanol/' + str(calib) + '/Tests/00_7p5C.txt')
etoh_calib_mid = np.loadtxt(project_folder + '/Processed/Ethanol/' + str(calib) + '/Tests/04_31p7C.txt')
etoh_calib_high = np.loadtxt(project_folder + '/Processed/Ethanol/' + str(calib) + '/Tests/09_60p95C.txt')

plt.figure()
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[6]][0], linestyle='-', color=(0,0,1), linewidth=2.0, label='IJ 7.6°C')
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[6]][6], linestyle='-.', color=(0.5,0,0.5), linewidth=2.0, label='IJ 31.9°C')
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[6]][-1], linestyle=':', color=(1,0,0), linewidth=2.0, label='IJ 58.7°C')
plt.legend()
plt.xlabel('q (Å$^{-1}$)')
plt.ylabel('Intensity (arb. units)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.gca().set_ylim([0.3, 1.02])
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.title('Select Curves at y = 1.75 mm')
plt.tight_layout()
plt.savefig(project_folder + '/Processed/Ethanol/IJ Ramping/y175_comparison.png')
np.savetxt(project_folder + '/Processed/Ethanol/IJ Ramping/y1.75/y1p75_data.txt', np.transpose([reduced_q, [reduced_intensity[z] for z in positions[6]][0], 
            [reduced_intensity[z] for z in positions[6]][6], [reduced_intensity[z] for z in positions[6]][-1]]), header='q 7.6C 31.9C 58.7C',
            delimiter='\t')

plt.figure()
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[2]][0], linestyle='-', color=(0,0,1), linewidth=2.0, label='IJ 7.1°C')
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[2]][6], linestyle='-.', color=(0.5,0,0.5), linewidth=2.0, label='IJ 31.0°C')
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[2]][-1], linestyle=':', color=(1,0,0), linewidth=2.0, label='IJ 57.9°C')
plt.legend()
plt.xlabel('q (Å$^{-1}$)')
plt.ylabel('Intensity (arb. units)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.gca().set_ylim([0.3, 1.02])
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.title('Select Curves at y = 0.75 mm')
plt.tight_layout()
plt.savefig(project_folder + '/Processed/Ethanol/IJ Ramping/y075_comparison.png')
np.savetxt(project_folder + '/Processed/Ethanol/IJ Ramping/y0.75/y0p75_data.txt', np.transpose([reduced_q, [reduced_intensity[z] for z in positions[2]][0], 
            [reduced_intensity[z] for z in positions[2]][6], [reduced_intensity[z] for z in positions[2]][-1]]), header='q 7.1C 31.0C 57.9C',
            delimiter='\t')


plt.figure()
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[-1]][0], linestyle='-', color=(0,0,1), linewidth=2.0, label='IJ 9.3°C')
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[-1]][6], linestyle='-.', color=(0.5,0,0.5), linewidth=2.0, label='IJ 34.4°C')
plt.plot(reduced_q, [reduced_intensity[z] for z in positions[-1]][-1], linestyle=':', color=(1,0,0), linewidth=2.0, label='IJ 60.4°C')
plt.legend()
plt.xlabel('q (Å$^{-1}$)')
plt.ylabel('Intensity (arb. units)')
plt.autoscale(enable=True, axis='x', tight=True)
plt.gca().set_ylim([0.3, 1.02])
plt.minorticks_on()
plt.tick_params(which='both',direction='in')
plt.title('Select Curves at y = 10.0 mm')
plt.tight_layout()
plt.savefig(project_folder + '/Processed/Ethanol/IJ Ramping/y10_comparison.png')
np.savetxt(project_folder + '/Processed/Ethanol/IJ Ramping/y10.0/y10p0_data.txt', np.transpose([reduced_q, [reduced_intensity[z] for z in positions[-1]][0], 
            [reduced_intensity[z] for z in positions[-1]][6], [reduced_intensity[z] for z in positions[-1]][-1]]), header='q 9.3C 34.4C 60.4C',
            delimiter='\t')






