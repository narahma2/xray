"""
Processes the APS 2017-2 temperature data sets.
Creates the calibration sets to be used for the impinging jets.

Created on Sat Aug 25 14:51:24 2018

@author: rahmann
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.constants import convert_temperature
from general.xray_factor import ItoS
from temperature_processing import main as temperature_processing


def main(test, scan):
    prj_fld = '/mnt/r/X-ray Temperature/APS 2017-2/'
    fld = '{0}/Processed/{1}'.format(prj_fld, test)

    # Load scan
    f = h5py.File('{0}/RawData/Scan_{1}.hdf5'.format(prj_fld, scan), 'r')

    # Read in temperature
    T = list(f['7bm_dau1:dau:010:ADC'])
    T.append(T[-1])

    # Convert temperature from Celsius to Kelvin
    T = convert_temperature(T, 'Celsius', 'Kelvin')

    # Read in q values
    q = list(f['q'])

    # 2017 Water
    if scan in [397, 398]:
        g = h5py.File(prj_fld + '/RawData/Scan_402.hdf5', 'r')
        bg = [
              g['Intensity_vs_q'][:, i]
              for i in range(np.shape(g['Intensity_vs_q'])[1])
             ]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [
                         f['Intensity_vs_q'][:, i]
                         for i in range(np.shape(f['Intensity_vs_q'])[1])
                         ]
        sl = slice(
                   (np.abs(np.array(q) - 1.70)).argmin(),
                   (np.abs(np.array(q) - 3.1)).argmin()
                   )
        avg_rows = 12

    if scan in [400, 401, 403]:
        g = h5py.File(prj_fld + '/RawData/Scan_402.hdf5', 'r')
        bg = [
              g['Intensity_vs_q'][:, i]
              for i in range(np.shape(g['Intensity_vs_q'])[1])
              ]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [
                         f['Intensity_vs_q'][:, i]
                         for i in range(np.shape(f['Intensity_vs_q'])[1])
                         ]
        sl = slice(
                   (np.abs(np.array(q) - 1.70)).argmin(),
                   (np.abs(np.array(q) - 3.1)).argmin()
                   )
        avg_rows = 20

    # 2017 Ethanol
    # 404 looked at the same q range as the water scans
    if scan == 404:
        g = h5py.File(prj_fld + '/RawData/Scan_402.hdf5', 'r')
        bg = [
              g['Intensity_vs_q'][:, i]
              for i in range(np.shape(g['Intensity_vs_q'])[1])
              ]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [
                         f['Intensity_vs_q'][:, i]
                         for i in range(np.shape(f['Intensity_vs_q'])[1])
                         ]
        sl = slice(
                   (np.abs(np.array(q) - 1.70)).argmin(),
                   (np.abs(np.array(q) - 3.1)).argmin()
                   )
        avg_rows = 20

    # 408 and 409 had a different detector position (different q range)
    if scan == 408:
        g = h5py.File(prj_fld + '/RawData/Scan_410.hdf5', 'r')
        bg = [
              g['Intensity_vs_q'][:, i]
              for i in range(np.shape(g['Intensity_vs_q'])[1])
              ]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [
                         f['Intensity_vs_q'][:, i]
                         for i in range(np.shape(f['Intensity_vs_q'])[1])
                         ]
        sl = slice(
                   (np.abs(np.array(q) - 0.6)).argmin(),
                   (np.abs(np.array(q) - 1.75)).argmin()
                   )
        avg_rows = 10

    if scan == 409:
        g = h5py.File(prj_fld + '/RawData/Scan_410.hdf5', 'r')
        bg = [
              g['Intensity_vs_q'][:, i]
              for i in range(np.shape(g['Intensity_vs_q'])[1])
              ]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [
                         f['Intensity_vs_q'][:, i]
                         for i in range(np.shape(f['Intensity_vs_q'])[1])
                         ]
        sl = slice(
                   (np.abs(np.array(q) - 0.6)).argmin(),
                   (np.abs(np.array(q) - 1.75)).argmin()
                   )
        avg_rows = 10

    # 2017 Dodecane
    if scan == 414:
        g = h5py.File(prj_fld + '/RawData/Scan_416.hdf5', 'r')
        bg = [
              g['Intensity_vs_q'][:, i]
              for i in range(np.shape(g['Intensity_vs_q'])[1])
              ]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [
                         f['Intensity_vs_q'][:, i]
                         for i in range(np.shape(f['Intensity_vs_q'])[1])
                         ]
        sl = slice(
                   (np.abs(np.array(q) - 0.6)).argmin(),
                   (np.abs(np.array(q) - 1.75)).argmin()
                   )
        avg_rows = 10

    if scan == 415:
        g = h5py.File(prj_fld + '/RawData/Scan_416.hdf5', 'r')
        bg = [
              g['Intensity_vs_q'][:, i]
              for i in range(np.shape(g['Intensity_vs_q'])[1])
              ]
        bg_avg = np.mean(bg, axis=0)
        raw_intensity = [
                         f['Intensity_vs_q'][:, i]
                         for i in range(np.shape(f['Intensity_vs_q'])[1])
                         ]
        sl = slice(
                   (np.abs(np.array(q) - 0.6)).argmin(),
                   (np.abs(np.array(q) - 1.75)).argmin()
                   )
        avg_rows = 10

    # Background subtraction
    intensity = [(x-bg_avg) for x in raw_intensity]
    # intensity = raw_intensity

    # Intensity correction
    intensity_avg = []
    T_avg = []
    scatter = []

    # Bin the data sets
    for i in range(0, len(intensity) // avg_rows):
        start = i*avg_rows
        stop = ((i+1)*avg_rows)-1
        scatter.append(np.mean(f['Scatter_images'][start:stop], axis=0))
        intensity_avg.append(np.mean(intensity[start:stop], axis=0))
        T_avg.append(np.mean(T[start:stop]))

    i = (len(intensity) // avg_rows)
    if np.mod(len(intensity), avg_rows) != 0:
        start = i*avg_rows
        scatter.append(np.mean(f['Scatter_images'][start:-1], axis=0))
        intensity_avg.append(np.mean(intensity[start:-1], axis=0))
        T_avg.append(np.mean(T[start:-1]))

    # Savitzky-Golay filtering
    filt_I = [savgol_filter(x, 55, 3) for x in intensity_avg]

    reduced_q = np.array(q[sl])
    reduced_I = [x[sl] for x in filt_I]
    reduced_I = np.array([y/np.trapz(y, x=reduced_q) for y in reduced_I])

    if test == 'Water':
        sf = np.array([ItoS(np.array(reduced_q), x) for x in reduced_I])
    else:
        sf = None

    temperature_processing(test, fld, scan, reduced_I, reduced_q, T_avg, sf,
                           scatter=scatter, background=g['Scatter_images'])

    rr = np.array([(x-min(T_avg))/(max(T_avg)-min(T_avg)) for x in T_avg])
    bb = np.array([1-(x-min(T_avg))/(max(T_avg)-min(T_avg)) for x in T_avg])

    plots_folder = fld + '/' + str(scan) + '/Plots/'

    # Grab middle slice as median
    mid_temp = len(reduced_I) // 2
    plt.figure()
    plt.plot(
             reduced_q,
             reduced_I[0],
             linestyle='-',
             color=(rr[0], 0, bb[0]),
             linewidth=2.0,
             label='{0:0d} K'.format(T_avg[0])
             )
    plt.plot(
             reduced_q,
             reduced_I[mid_temp],
             linestyle='-.',
             color=(0.5, 0, 0.5),
             linewidth=2.0,
             label='{0:0d} K'.format(T_avg[mid_temp])
             )
    plt.plot(
             reduced_q,
             reduced_I[-1],
             linestyle=':',
             color=(rr[-1], 0, bb[-1]),
             linewidth=2.0,
             label='{0:0d} K'.format(T_avg[-1])
             )
    plt.legend()
    plt.xlabel('q (Å$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in')
    plt.title('Select ' + test + ' Curves')
    plt.tight_layout()
    plt.savefig(plots_folder + 'selectcurves.png')
    plt.close()


def run_main():
    # Water: 400, 401, 403
    # Ethanol: 404, 408, 409
    # Dodecane: 414, 415
    scans = [
             400, 401, 403,
             408, 409,
             414, 415
             ]
    tests = [
             'Water', 'Water', 'Water',
             'Ethanol', 'Ethanol',
             'Dodecane', 'Dodecane'
             ]

    # Run all tests
    [main(tests[i], scans[i]) for i, _ in enumerate(scans)]

    # Run singular test
    # main('Ethanol', 409)

    # Run select tests
    # [
    #  main(tests[i], scans[i])
    #  for i in [
    #            j
    #            for j,_ in enumerate(tests) if tests[j] == 'Ethanol'
    #            ]
    #  ]


if __name__ == '__main__':
    run_main()
