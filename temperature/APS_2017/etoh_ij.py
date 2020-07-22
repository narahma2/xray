"""
Created on Wed May  8 14:29:08 2019

@author: rahmann
"""

import h5py
import numpy as np
from scipy.signal import savgol_filter
from temperature_processing import main as temperature_processing


def main():
    prj_fld = '/mnt/r/X-ray Temperature/APS 2017-2'

    tests = ['Ethanol/IJ Ambient', 'Ethanol/IJ Hot']

    for test in tests:
        folder = prj_fld + '/Processed/Ethanol'

        if 'Ambient' in test:
            scans = [438, 439, 440, 441, 442, 443, 444]
            bg_scan = 437
            y = [0.1, 1, 2, 2.5, 4, 15, 25]
        if 'Hot' in test:
            scans = [428, 430, 431]
            bg_scan = 429
            # Y position for Scan 430 is faked
            y = [10, 15, 25]

        g = h5py.File(
                      '{0}/RawData/Scan_{1}.hdf5'.format(prj_fld, bg_scan),
                      'r'
                      )
        bg = [
              g['Intensity_vs_q'][:, i]
              for i in range(np.shape(g['Intensity_vs_q'])[1])
              ]
        bg_avg = np.mean(bg, axis=0)

        intensities = []
        scatter = []
        for n, scan in enumerate(scans):
            f = h5py.File(
                          '{0}/RawData/Scan_{1}.hdf5'.format(prj_fld, scan),
                          'r'
                          )
            q = list(f['q'])
            intensity = [
                         f['Intensity_vs_q'][:, i]
                         for i in range(np.shape(f['Intensity_vs_q'])[1])
                         ]
            intensities.append(np.mean(intensity, axis=0))
            scatter.append(f['Scatter_images'][n])

        sl = slice(
                   (np.abs(np.array(q) - 0.6)).argmin(),
                   (np.abs(np.array(q) - 1.75)).argmin()
                   )
        intensities = [(x-bg_avg) for x in intensities]
        filtered_I = [savgol_filter(x, 55, 3) for x in intensities]
        reduced_q = np.array(q[sl])
        reduced_I = [x[sl] for x in filtered_I]
        #reduced_I = [y/np.trapz(y, x=reduced_q) for y in reduced_I]

        temperature_processing(test.rsplit('/')[0], folder,
                               test.rsplit('/')[1], reduced_I, reduced_q,
                               temperature=None, structure_factor=None, y=y,
                               ramping=False, scatter=scatter,
                               background=g['Scatter_images'])


if __name__ == '__main__':
    main()
