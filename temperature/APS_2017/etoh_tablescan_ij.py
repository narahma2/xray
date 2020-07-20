"""
Processes the ethanol impinging jet table scan (425).
See "X-ray Temperature/APS 2017-2/IJ Ethanol Ramping" in OneNote.

Created on Sat Apr  6 10:15:38 2019

@author: rahmann
"""

import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.constants import convert_temperature
from scipy.signal import savgol_filter
from temperature_processing import main as temperature_processing


def main():
    test = 'Ethanol/IJ Ramping'
    rank_x = np.linspace(0, 13, num=14, dtype=int)

    prj_fld = '/mnt/r/X-ray Temperature/APS 2017-2'
    folder = prj_fld + '/Processed/Ethanol'

    # Load in intensities
    f = h5py.File(prj_fld + '/RawData/Scan_' + str(425) + '.hdf5', 'r')

    q = list(f['q'])
    intensities = [
                   f['Intensity_vs_q'][:, i]
                   for i in range(np.shape(f['Intensity_vs_q'])[1])
                   ]

    sl = slice(
               (np.abs(np.array(q) - 0.6)).argmin(),
               (np.abs(np.array(q) - 1.75)).argmin()
               )

    nozzle_T = []
    y_loc = []

    for i in rank_x:
        y_locs = list(f['Rank_2_Point_' + str(i) + '/7bmb1:aero:m1.VAL'])
        for n, y in enumerate(y_locs):
            nozzle_T.append(list(f[
                                   'Rank_2_Point_{0}/7bm_dau1:dau:010:ADC'
                                   .format(i)
                                   ])[n])
            y_loc.append(round(list(f[
                                      'Rank_2_Point_{0}/7bmb1:aero:m1.VAL'
                                      .format(i)
                                      ])[n], 2))

    # Convert temperatures to numpy array and from Celsius to Kelvin
    nozzle_T = np.array(nozzle_T)
    nozzle_T = convert_temperature(nozzle_T, 'Celsius', 'Kelvin')

    positions = []
    for k in y_locs:
        # 14 total occurrences of the 18 y positions
        positions.append([i for i, x in enumerate(y_loc) if x == round(k, 2)])

    # Read only the first 252 intensity scans
    # There's 277, seems to be extraneous data?
    intensities = np.array(intensities[0:252])

    # Load in background
    g = h5py.File(prj_fld + '/RawData/Scan_' + str(429) + '.hdf5', 'r')

    for x in positions[0]:
        # Make the y = 10.0 mm position the background
        bg = intensities[x+17]
        # Background subtraction
        intensities[x:(x+18)] = intensities[x:(x+18)] - bg

    # Smooth out data using Savitzky-Golay filter
    filtered_intensity = [savgol_filter(x, 55, 3) for x in intensities]

    # Crop q range to look at same window as EtOH calibration sets
    reduced_q = np.array(q[sl])
    reduced_intensity = [x[sl] for x in filtered_intensity]

    # Normalize intensity by the area (same process as the Nature paper)
    reduced_intensity = [
                         y/np.trapz(y, x=reduced_q)
                         for y in reduced_intensity
                         ]

    # Convert reduced_intensity to numpy array
    reduced_intensity = np.array(reduced_intensity)

    # Constant temperature plots (variable position)
    for x in positions[0]:
        # Crop out y = 9.0 and 10.0 mm points
        y = y_loc[x:(x+16)]
        temperature = np.mean(nozzle_T[x:(x+16)])
        par_test = test.rsplit('/')[0]
        par_scan = (
                    test.rsplit('/')[1] + '/Temperature/T' +
                    '{0:05.2f}'.format(temperature).replace('.', 'p')
                    )
        temperature_processing(par_test, folder, par_scan,
                               reduced_intensity[x:(x+16)], reduced_q,
                               temperature=None, structure_factor=None, y=y,
                               ramping=True,
                               scatter=f['Scatter_images'][x:(x+16)],
                               background=g['Scatter_images'])

    # Constant position plots (variable temperature)
    for x, i in enumerate(positions[:-1]):
        y = y_locs[x]
        par_test = test.rsplit('/')[0]
        par_scan = (
                    test.rsplit('/')[1] + '/Positions/y' +
                    '{0:05.2f}'.format(y).replace('.', 'p')
                    )
        temperature_processing(par_test, folder, par_scan,
                               reduced_intensity[i], reduced_q,
                               temperature=nozzle_T[i],
                               structure_factor=None, y=None, ramping=True,
                               scatter=f['Scatter_images'][i],
                               background=g['Scatter_images'])

    # Plot calibration jets for each of the constant T cases
    # Load calibration data set
    with open(
              glob.glob(prj_fld + '/Processed/Ethanol/409/*.pckl')[0],
              'rb'
              ) as f:
        temperature_cal, reduced_q_cal, reduced_intensity_cal = pickle.load(f)

    flds = glob.glob(folder + '/IJ Ramping/Temperature/T*/')
    for fld in flds:
        # IJ temperature
        T = float(fld.split('/')[-2][1:].replace('p', '.'))

        # Load calibration for selected IJ temperature
        ind = np.argmin(abs(np.array(temperature_cal) - T))
        intensity_cal = reduced_intensity_cal[ind]

        # Load IJ Ramping data set
        with open(glob.glob(fld + '/*.pckl')[0], 'rb') as f:
            temperature, y, reduced_q, reduced_intensity = pickle.load(f)

        ij_y = []
        ij_I = []
        for i, _ in enumerate(y):
            if not np.mod(i, 3):
                ij_y.append(y[i])
                ij_I.append(reduced_intensity[i])

        # Plot combined scans
        plt.figure()
        [
         plt.plot(
                  reduced_q,
                  ij_I[i],
                  color='C' + str(i),
                  linestyle='--',
                  label='y = {0:0.2f}'.format(ij_y[i])
                  )
         for i, _ in enumerate(ij_I)
         ]
        plt.plot(
                 reduced_q_cal,
                 intensity_cal,
                 color='k',
                 linestyle='-',
                 label='Calib. Jet'
                 )
        plt.title('T = ' + str(round(T)) + ' K')
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('Intensity (arb. units)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fld + '/combined.png')
        plt.close()


if __name__ == '__main__':
    main()
