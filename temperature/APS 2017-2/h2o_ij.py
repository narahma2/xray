# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:02:13 2019

@author: rahmann
"""

import sys
if sys.platform == 'win32':
    gh_fld = 'E:/GitHub/xray/general'
    sys.path.append(gh_fld)
    sys.path.append('E:/GitHub/xray/temperature')
    sys_folder = 'R:'
elif sys.platform == 'linux':
    gh_fld = '/mnt/e/GitHub/xray/general'
    sys.path.append(gh_fld)
    sys.path.append('/mnt/e/GitHub/xray/temperature')
    sys_folder = '/mnt/r/'

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(gh_fld + '/python/matplotlib/stylelib/paper.mplstyle')
import pickle
from scipy.constants import convert_temperature
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from calc_statistics import comparefit
from temperature_processing import main as temperature_processing

#%% Plotting functions
def calibrate(folder, calib_folder, name, ij_mapping_T, profile, x_loc):
	calib = np.poly1d(np.loadtxt(calib_folder + '/' + name + '_polynomial.txt'))
	interpolate(x_loc, calib, profile, name, folder)
	ij_mapping_T.setdefault(name,[]).append(calib(profile))
	return profile, ij_mapping_T

def interpolate(x_loc, calib, profile, name, folder):
	plt.figure()
	plt.plot(x_loc, profile, ' o', markerfacecolor='none', markeredgecolor='b')
	plt.title('Temperature from ' + name)
	plt.ylabel(name)
	plt.xlabel('X Location (mm)')
	plt.autoscale(enable=True, axis='x', tight=True)
	plt.minorticks_on()
	plt.tick_params(which='both',direction='in')
	plt.tight_layout()
	plt.savefig(folder + '/Statistics/' + name +'.png')
	plt.close()
	
	plt.figure()
	plt.plot(x_loc, calib(profile), ' o', markerfacecolor='none', markeredgecolor='b')
	plt.title('Temperature from ' + name)
	plt.ylabel('Temperature (K)')
	plt.xlabel('X Location (mm)')
	plt.autoscale(enable=True, axis='x', tight=True)
	plt.minorticks_on()
	plt.tick_params(which='both',direction='in')
	plt.tight_layout()
	plt.savefig(folder + '/Temperature/' + name +'.png')
	plt.close()

#%%
project_folder = sys_folder + '/X-ray Temperature/APS 2017-2'

test = 'Water/IJ Mixing'
scan_no = [452, 453, 448, 446, 447, 454, 455, 456]
y_location = [0.5, 0.5, 1.5, 2, 2.5, 3.0, 5.0, 10.0]

g = h5py.File(project_folder + '/RawData/Scan_437.hdf5', 'r')
bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
bg_avg = np.mean(bg, axis=0)

calib_folder = project_folder + '/Processed/Water/403/Statistics'
# calib_folder = sys_folder + '/X-ray Temperature/APS 2018-1/Processed/Water_700umNozzle/Combined/Statistics'

ij_mapping_X = []
ij_mapping_Y = []
ij_mapping_T = {}

for i, scan in enumerate(scan_no):
	y_loc = y_location[i]
	folder = project_folder + '/Processed/Water/IJ Mixing/Scan' + str(scan)
	if not os.path.exists(folder + '/Curves'):
		os.makedirs(folder + '/Curves')
	if not os.path.exists(folder + '/Temperature'):
		os.makedirs(folder + '/Temperature')
	if not os.path.exists(folder + '/Statistics'):
		os.makedirs(folder + '/Statistics')
	
	f = h5py.File(project_folder + '/RawData/Scan_' + str(scan) + '.hdf5', 'r')
	x_loc = list(f['7bmb1:aero:m2.VAL'])
	q = list(f['q'])
	sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
	pinned_sl = np.abs(np.array(q) - 2.78).argmin()
	intensity = [f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])]
	intensity = [(x-bg_avg) for x in intensity]
	filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]
	reduced_q = q[sl]
	reduced_intensity = [x[sl] for x in filtered_intensity]
	reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]
	concavity = [-np.gradient(np.gradient(i)) for i in reduced_intensity]
	concavity = [savgol_filter(j, 55, 3) for j in concavity]
	peak_locs = [find_peaks(k, height=0.00001, distance=100)[0] for k in concavity]
	
	peakq, ij_mapping_T = calibrate(folder, calib_folder, 'peakq', ij_mapping_T, np.array([reduced_q[x[0]] for x in peak_locs]), x_loc)
	# q2, ij_mapping_T = calibrate(folder, calib_folder, 'q2', ij_mapping_T, np.array([reduced_q[x[1]] if len(x) == 2 else np.nan for x in peak_locs]), x_loc)
	_, ij_mapping_T = calibrate(folder, calib_folder, 'aratio', ij_mapping_T, [np.trapz(x[:pinned_sl], reduced_q[:pinned_sl]) / np.trapz(x[pinned_sl:], reduced_q[pinned_sl:]) for x in reduced_intensity], x_loc)
	_, ij_mapping_T = calibrate(folder, calib_folder, 'var', ij_mapping_T, [stats.skew(k) for k in reduced_intensity], x_loc)
	_, ij_mapping_T = calibrate(folder, calib_folder, 'skew', ij_mapping_T, [np.var(k) for k in reduced_intensity], x_loc)
	_, ij_mapping_T = calibrate(folder, calib_folder, 'kurt', ij_mapping_T, [stats.kurtosis(k) for k in reduced_intensity], x_loc)
	
		
	for j,_ in enumerate(reduced_intensity):
		plt.figure()
		plt.plot(reduced_q, reduced_intensity[j], color='k', linewidth=2.0, label='x = ' + str(round(x_loc[j],2)) + ' mm')
		plt.axvline(x=peakq[j], linestyle='--', color='C0', label='peakq = ' + str(round(peakq[j], 2)))
		# plt.axvline(x=q2[j], linestyle='--', color='C1', label='q2 = ' + str(round(q2[j], 2)))
		plt.legend()
		plt.xlabel('q (Å$^{-1}$)')
		plt.ylabel('Intensity (arb. units)')
		plt.autoscale(enable=True, axis='x', tight=True)
		plt.gca().set_ylim([0, 1.02])
		plt.minorticks_on()
		plt.tick_params(which='both',direction='in')
		plt.title('Scan ' + str(scan))
		plt.tight_layout()
		plt.savefig(folder + '/Curves/' + str(j).zfill(3) + '.png')
		plt.close()
	
	ij_mapping_X.append(x_loc)
	ij_mapping_Y.append(y_loc)
	
	with open(folder + '/' + str(scan) + '_data.pckl', 'wb') as f:
		pickle.dump([x_loc, reduced_q, reduced_intensity, peakq], f)
	
#%% Create density map of temperature
xx = np.linspace(-2,2,81)
yy = np.linspace(0, 12, 25)
xv, yv = np.meshgrid(xx, yy)
keys = list(ij_mapping_T.keys())
zv = {key: np.zeros(xv.shape) for key in keys}

for ycount, _ in enumerate(ij_mapping_Y):
	jj = np.sum(np.square(np.abs(yv-ij_mapping_Y[ycount])),1).argmin()
	for xcount, _ in enumerate(ij_mapping_X[ycount]):
		for key in ij_mapping_T.keys():
			ii = np.abs(xv-ij_mapping_X[ycount][xcount]).argmin()
			zv[key][jj,ii] = ij_mapping_T[key][ycount][xcount]
			if not (0 <= zv[key][jj,ii] <= 100):
				zv[key][jj,ii] = 0

def temperature_plot(T, Ttype):    
	fig, ax = plt.subplots()
	pc = ax.pcolormesh(xv, yv, T, cmap=plt.cm.bwr)
	cbar = fig.colorbar(pc)
	cbar.set_label('Temperature (C)')
	plt.gca().invert_yaxis()
	plt.xlabel('X Position (mm)')
	plt.ylabel('Y Position (mm)')
	plt.title(Ttype)
	
	def format_coord(xx, yy):
		xarr = xv[0,:]
		yarr = yv[:,0]
		if ((xx > xarr.min()) & (xx <= xarr.max()) & 
			(yy > yarr.min()) & (yy <= yarr.max())):
			col = np.searchsorted(xarr, xx)-1
			row = np.searchsorted(yarr, yy)-1
			zz = T[row, col]
			return f'x={xx:1.4f}, y={yy:1.4f}, z={zz:1.4f}   [{row},{col}]'
		else:
			return f'x={xx:1.4f}, y={yy:1.4f}'
	
	ax.format_coord = format_coord
	
	plt.show()
	plt.tight_layout()
	plt.savefig(folder.rsplit('Scan')[0] + Ttype + '.png')
	plt.close()

[temperature_plot(zv[x], x) for x in keys]
	
	
	
	
	
	
	
	
	
	
