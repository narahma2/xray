"""
Created on Fri Apr  5 01:02:13 2019

@author: rahmann
"""

import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from copy import copy
from datetime import datetime
from scipy import spatial
from scipy.constants import convert_temperature
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from temperature_processing import saveimage, main as temperature_processing, pca

# Plotting functions
def calibrate(folder, calib_folder, name, ij_mapping_T, profile, x_loc, cal_I=None, cal_T=None):
    if name == 'KDTree':
        tree = spatial.KDTree(cal_I)
        (calib, position) = [tree.query(x) for x in profile]
        temperature = np.array(cal_T)[position]
        ij_mapping_T.setdefault(name,[]).append(temperature)
    else:
        calib = np.poly1d(np.loadtxt(calib_folder + '/Statistics/' + name + '_polynomial.txt'))
        ij_mapping_T.setdefault(name,[]).append(calib(profile))

    interpolate(x_loc, calib, profile, name, folder)
    return profile, ij_mapping_T


def interpolate(x_loc, calib, profile, name, folder):
    plt.figure()
    plt.plot(x_loc, profile, ' o', markerfacecolor='none', markeredgecolor='b')
    plt.title(name)
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
    axes = plt.gca()
    axes.set_ylim([250, 320])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()
    plt.tick_params(which='both',direction='in')
    plt.tight_layout()
    plt.savefig(folder + '/Temperature/' + name +'.png')
    plt.close()


def main():
    project_folder = '/mnt/r/X-ray Temperature/APS 2017-2'

    test = 'Water/IJ Mixing'
    scan_no = [452, 453, 448, 446, 447, 454, 455, 456]
    y_location = [0.5, 0.5, 1.5, 2, 2.5, 3.0, 5.0, 10.0]

    g = h5py.File(project_folder + '/RawData/Scan_437.hdf5', 'r')
    bg = [g['Intensity_vs_q'][:,i] for i in range(np.shape(g['Intensity_vs_q'])[1])]
    bg_avg = np.mean(bg, axis=0)

    calib_folders = [project_folder + '/Processed/Water/403', '/mnt/r/X-ray Temperature/APS 2018-1/Processed/Water_700umNozzle/Combined']
    calibrations = ['403', 'Combined'] 

    ij_mapping_X = []
    ij_mapping_Y = []
    ij_mapping_T = {}

    for n, calib_folder in enumerate(calib_folders):
        calib = calibrations[n]
        with open(glob.glob(calib_folder + '/*.pckl')[0], 'rb') as f:
            temperature_cal, reduced_q_cal, reduced_intensity_cal = pickle.load(f)

        # Find locations of the cold, mid, and hot temperatures in the calibration set
        sl_cold = np.argmin(abs(np.array(temperature_cal) - 276))
        sl_mid = np.argmin(abs(np.array(temperature_cal) - 287))
        sl_hot = np.argmin(abs(np.array(temperature_cal) - 298))

        for i, scan in enumerate(scan_no):
            y_loc = y_location[i]
            folder = project_folder + '/Processed/Water/IJ Mixing/' + calib + '/y' + '{0:05.2f}'.format(y_location[i]).replace('.', 'p') + '_' + 'scan' + str(scan)
            if not os.path.exists(folder + '/Curves'):
                os.makedirs(folder + '/Curves')
            if not os.path.exists(folder + '/Temperature'):
                os.makedirs(folder + '/Temperature')
            if not os.path.exists(folder + '/Statistics'):
                os.makedirs(folder + '/Statistics')
            if not os.path.exists(folder + '/Images'):
                os.makedirs(folder + '/Images')
            
            f = h5py.File(project_folder + '/RawData/Scan_' + str(scan) + '.hdf5', 'r')
            x_loc = np.array(list(f['7bmb1:aero:m2.VAL']))
            saveimage(folder + '/Images', x_loc, f['Scatter_images'], g['Scatter_images'])
            q = list(f['q'])
            sl = slice((np.abs(np.array(q) - 1.70)).argmin(), (np.abs(np.array(q) - 3.1)).argmin())
            pinned_sl = np.abs(np.array(q) - 2.78).argmin()
            intensity = np.array([f['Intensity_vs_q'][:,i] for i in range(np.shape(f['Intensity_vs_q'])[1])])

            # Let the first point be the background
            for b, w in enumerate(intensity[1:], start=1):
                if np.sum(w) > 510000:
                    intensity[b] = w - intensity[0]
                else:
                    intensity[b] = w * 0

            # Let Scan 437 be the background
            # intensity = [(x-bg_avg) for x in intensity]

            filtered_intensity = [savgol_filter(x, 55, 3) for x in intensity]
            reduced_q = q[sl]
            reduced_intensity = [x[sl] for x in filtered_intensity]
            #reduced_intensity = [y/np.trapz(y, x=reduced_q) for y in reduced_intensity]

            # Filter out NaN
            nan_ind = [~np.isnan(x[0]) for x in reduced_intensity]
            reduced_intensity = np.array(reduced_intensity)[nan_ind]
            x_loc = x_loc[nan_ind]

            peak_locs = [np.argmax(x) for x in reduced_intensity]

            # Create profiles
            # _, ij_mapping_T = calibrate(folder, calib_folder, 'KDTree', ij_mapping_T, np.array([np.interp(reduced_q_cal, reduced_q, x) for x in reduced_intensity]), x_loc, cal_I=reduced_intensity_cal, cal_T=temperature_cal)
            peak, ij_mapping_T = calibrate(folder, calib_folder, 'peak', ij_mapping_T, np.array([reduced_intensity[n][x] for n, x in enumerate(peak_locs)]), x_loc)
            peakq, ij_mapping_T = calibrate(folder, calib_folder, 'peakq', ij_mapping_T, np.array([reduced_q[x] for x in peak_locs]), x_loc)
            _, ij_mapping_T = calibrate(folder, calib_folder, 'pca', ij_mapping_T, pca(reduced_intensity), x_loc)
            _, ij_mapping_T = calibrate(folder, calib_folder, 'aratio', ij_mapping_T, [np.trapz(x[:pinned_sl], reduced_q[:pinned_sl]) / np.trapz(x[pinned_sl:], reduced_q[pinned_sl:]) for x in reduced_intensity], x_loc)
            _, ij_mapping_T = calibrate(folder, calib_folder, 'var', ij_mapping_T, [stats.skew(k) for k in reduced_intensity], x_loc)
            _, ij_mapping_T = calibrate(folder, calib_folder, 'skew', ij_mapping_T, [np.var(k) for k in reduced_intensity], x_loc)
            _, ij_mapping_T = calibrate(folder, calib_folder, 'kurt', ij_mapping_T, [stats.kurtosis(k) for k in reduced_intensity], x_loc)
            
            # Plot intensity curves
            for j,_ in enumerate(reduced_intensity):
                plt.figure()
                plt.plot(reduced_q, reduced_intensity[j], color='k', linewidth=2.0, label='x = ' + str(round(x_loc[j],2)) + ' mm')
                plt.plot(reduced_q_cal, reduced_intensity_cal[sl_cold], color=(0,0,1), linestyle='--', linewidth=2.0, label='Calib. @ 276 K')
                plt.plot(reduced_q_cal, reduced_intensity_cal[sl_mid], color=(0.5,0,0.5), linestyle='--', linewidth=2.0, label='Calib. @ 287 K')
                plt.plot(reduced_q_cal, reduced_intensity_cal[sl_hot], color=(1,0,0), linestyle='--', linewidth=2.0, label='Calib. @ 298 K')
                plt.axvline(x=peakq[j], linestyle='--', color='C0', label='peakq = ' + str(round(peakq[j], 2)))
                # plt.axvline(x=q2[j], linestyle='--', color='C1', label='q2 = ' + str(round(q2[j], 2)))
                plt.legend()
                plt.xlabel('q (Å$^{-1}$)')
                plt.ylabel('Intensity (arb. units)')
                plt.autoscale(enable=True, axis='x', tight=True)
                # plt.gca().set_ylim([0, 1.02])
                plt.minorticks_on()
                plt.tick_params(which='both',direction='in')
                plt.title(str(y_location[i]) + ' mm')
                plt.tight_layout()
                plt.savefig(folder + '/Curves/' + str(j).zfill(3) + '_' + ('%06.2f'%x_loc[j]).replace('.', 'p') +  '.png')
                plt.close()
            
            ij_mapping_X.append(x_loc)
            ij_mapping_Y.append(y_loc)
            
            with open(folder + '/' + str(scan) + '_data.pckl', 'wb') as f:
                pickle.dump([x_loc, reduced_q, reduced_intensity, peakq], f)
            with open(folder + '/' + str(scan) + '_log.txt', 'a+') as f:
                f.write(datetime.now().strftime("\n%d-%b-%Y %I:%M:%S %p"))
        
        #%% Create density map of temperature
        xx = np.linspace(-2,2,81)
        yy = np.linspace(0, 12, 25)
        xv, yv = np.meshgrid(xx, yy)
        keys = list(ij_mapping_T.keys())
        zv = {key: np.zeros(xv.shape)*np.nan for key in keys}

        for ycount, _ in enumerate(ij_mapping_Y):
            jj = np.sum(np.square(np.abs(yv-ij_mapping_Y[ycount])),1).argmin()
            for xcount, _ in enumerate(ij_mapping_X[ycount]):
                for key in ij_mapping_T.keys():
                    ii = np.abs(xv-ij_mapping_X[ycount][xcount]).argmin()
                    zv[key][jj,ii] = ij_mapping_T[key][ycount][xcount]
                    if not (250 <= zv[key][jj,ii] <= 320):
                        zv[key][jj,ii] = np.nan

        def temperature_plot(T, Ttype):    
            fig, ax = plt.subplots()
            palette = copy(plt.cm.bwr)
            palette.set_bad('k')
            pc = ax.pcolormesh(xv, yv, T, cmap=palette, vmin=270, vmax=300)
            cbar = fig.colorbar(pc)
            cbar.set_label('Temperature (K)')
            plt.gca().invert_yaxis()
            plt.xlabel('X Position (mm)')
            plt.ylabel('Y Position (mm)')
            plt.title(Ttype)
            

            def format_coord(xx, yy):
                xarr = xv[0,:]
                yarr = yv[:,0]
                if ((xx > xarr.min()) & (xx <= xarr.max()) & (yy > yarr.min()) & (yy <= yarr.max())):
                    col = np.searchsorted(xarr, xx)-1
                    row = np.searchsorted(yarr, yy)-1
                    zz = T[row, col]
                    return f'x={xx:1.4f}, y={yy:1.4f}, z={zz:1.4f}   [{row},{col}]'
                else:
                    return f'x={xx:1.4f}, y={yy:1.4f}'
            

            ax.format_coord = format_coord
            
            plt.show()
            plt.tight_layout()
            plt.savefig(folder.rsplit('/y')[0] + '/' + Ttype + '.png')
            plt.close()

        [temperature_plot(zv[x], x) for x in keys]
    

if __name__ == '__main__':
    main()
