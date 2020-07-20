"""
Functions for processing temperature data sets (calibration + IJ).

Created on Tue Jan 21 13:54:00 2020

@author: rahmann
"""

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from scipy.signal import find_peaks
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from general.calc_statistics import polyfit
from general.misc import create_folder


# Set project folder
prj_fld = '/mnt/r/X-ray Temperature/'


# Functions for intensity profile fitting
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


def poly4(x, c0, c1, c2, c3, c4):
    return c0*x**4 + c1*x**3 + c2*x**2 + c3*x + c4


def profile(name, fit_var, profile, prfl_fld, stats_fld,
            test, plt_fld):
    """
    Creates plots for the potential thermometer/positioning profiles.
    =============
    --VARIABLES--
    name:        Profile name ('peak', 'q1', etc.)
    fit_var:     Y axis variable (temperature, y location, or index)
    profile:     X axis variable ('peak', 'q1', etc.)
    prfl_fld:    Location where profiles (X axis values) will be saved.
    stats_fld:   Location where statistics (polynomial fit) will be saved.
    test:        Type of liquid ("Water", "Ethanol", "Dodecane")
    plt_fld      Location where plots will be saved.
    """

    np.savetxt('{0}/profile_{1}.txt'.format(prfl_fld, name), profile)

    # Do fitting and plots only for the temperature/y location folders
    # (not for the pooled IJ ramping case)
    if fit_var[-1] != 252:
        prfl_fit = polyfit(profile, fit_var, 1)
        np.savetxt(
                   '{0}/{1}_polynomial.txt'.format(stats_fld, name),
                   prfl_fit['polynomial']
                   )

        plt.figure()
        plt.plot(
                 profile,
                 fit_var,
                 ' o',
                 markerfacecolor='none',
                 markeredgecolor='b',
                 label='Data'
                 )
        plt.plot(
                 profile,
                 prfl_fit['function'](profile),
                 'k',
                 linewidth=2.0,
                 label='y = {0:0.2f}x + {1:0.2f}'
                       .format(prfl_fit['polynomial'][0],
                               prfl_fit['polynomial'][1])
                 )
        plt.title('{0} {1} - R$^2$ = {2:0.4f}'
                  .format(test, name, prfl_fit['determination']))
        plt.legend()
        if any(x in prfl_fld for x in ['IJ Ramping/Temperature',
                                       'IJ Ambient', 'IJ65', 'Cold']):
            plt.ylabel('Y Location (mm)')
        else:
            plt.ylabel('Temperature (K)')
        plt.xlabel(name)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.minorticks_on()
        plt.tick_params(which='both', direction='in')
        plt.tight_layout()
        plt.savefig('{0}/{1}.png'.format(plt_fld, name))
        plt.close()


def density(test, peak_q, T):
    """
    Computes density based on q spacing data. Serves as a measure of error.
    =============
    --VARIABLES--
    test:           Type of liquid: "Water", "Ethanol", "Dodecane"
    peak_q:         Location of peak in q [1/A]
    T:              Temperature of liquid [K]
    """

    # Convert q to d (q = 4*sin(theta)*pi/lambda; 2*d*sin(theta) = n*lambda)
    d = 2*np.pi/peak_q

    # Get temperatures and densities from EES data file
    density_file = '{0}/ScriptData/densities.txt'.format(prj_fld)
    density_data = pd.read_csv(density_file, sep="\t")
    temp = density_data["T_" + test[0:3].lower()]
    dens = density_data["rho" + test[0:3].lower()]

    return d, temp, dens


def saveimage(img_fld, fit_var, scatter, background):
    """
    Saves the scatter (w/o subtraction) and background images as 16 bit TIF.
    =============
    --VARIABLES--
    img_fld:            Save location of the TIF files.
    fit_var:            Temperature or y location variable.
    scatter:            Array containing scatter images.
    background:         Array containing background images.
    """

    # Average the background array
    background = np.mean(background, axis=0)

    # Save background
    Image.fromarray(background).save(img_fld + '/BG.tif')

    # Save images
    [
     Image.fromarray(x).save(
                             '{0}/{1:03d}_{2:06.2f}.tif'
                             .format(img_fld, n, fit_var[n])
                             .replace('.', 'p')
                             )
     for n, x in enumerate(scatter)
     ]


def pca(intensity):
    # StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(intensity)
    scaled_intensity = scaler.transform(intensity)

    # PCA
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(scaled_intensity)
    pc1 = principalComponents[:, 0]

    return pc1


def main(test, folder, scan, reduced_intensity, reduced_q,
         temperature=None, structure_factor=None, y=None, ramping=False,
         scatter=None, background=None, pooled=False):
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
    pooled              Ramping IJ pooled case (True/False)
    """

    prfl_fld = create_folder('{0}/{1}/Profiles/'.format(folder, scan))
    stats_fld = create_folder('{0}/{1}/Statistics/'.format(folder, scan))
    plt_fld = create_folder('{0}/{1}/Plots/'.format(folder, scan))
    tests_fld = create_folder('{0}/{1}/Tests/'.format(folder, scan))
    curves_fld = create_folder('{0}/{1}/Curves/'.format(folder, scan))

    if structure_factor is not None:
        sf_fld = create_folder('{0}/{1}/Structure Factor/'
                               .format(folder, scan))

    if 'IJ' in str(scan) and 'Ethanol' in test:
        pinned_pts = np.abs(reduced_q - 1.40).argmin()
    elif 'IJ' in str(scan) and 'Water' in test:
        pinned_pts = np.abs(reduced_q - 2.79).argmin()
    else:
        # Find pinned points in the curves (least variation)
        intensity_std = np.std(reduced_intensity, axis=0)
        pinned_pts = find_peaks(-intensity_std)[0]
        # Find the minimum peak only (throw away every other valley)
        pinned_pts = pinned_pts[np.argmin(intensity_std[pinned_pts])]

    pinned_q = reduced_q[pinned_pts]

    # Designate fit_var
    if pooled:
        index = np.linspace(1, len(reduced_intensity), len(reduced_intensity))
        fit_var = index
        data_label = ''
    else:
        if y is not None:
            fit_var = y
            data_label = ' mm'
            np.savetxt('{0}/positions.txt'.format(prfl_fld), y)
            np.savetxt('{0}/positions.txt'.format(stats_fld), y)
        elif temperature is not None:
            fit_var = temperature
            data_label = ' K'
            np.savetxt('{0}/temperature.txt'.format(prfl_fld), temperature)
            np.savetxt('{0}/temperature.txt'.format(stats_fld), temperature)

    # Save images if scatter and background arrays are passed
    if scatter is not None:
        img_fld = create_folder('{0}/{1}/Images/'.format(folder, scan))
        saveimage(img_fld, fit_var, scatter, background)

    # Save intensities in tests_fld
    [
     np.savetxt('{0}/{1:03.0f}.txt'.format(tests_fld, i), x)
     for i, x in enumerate(reduced_intensity)
     ]

    profile('peak', fit_var,
            [np.max(x) for x in reduced_intensity],
            prfl_fld, stats_fld, test, plt_fld)
    profile('peakq', fit_var,
            [reduced_q[np.argmax(x)] for x in reduced_intensity],
            prfl_fld, stats_fld, test, plt_fld)
    profile('aratio', fit_var,
            [
             np.trapz(x[:pinned_pts], reduced_q[:pinned_pts]) /
             np.trapz(x[pinned_pts:], reduced_q[pinned_pts:])
             for x in reduced_intensity
             ],
            prfl_fld, stats_fld, test, plt_fld)
    profile('mean', fit_var,
            [np.mean(x) for x in reduced_intensity],
            prfl_fld, stats_fld, test, plt_fld)
    profile('var', fit_var,
            [np.var(x) for x in reduced_intensity],
            prfl_fld, stats_fld, test, plt_fld)
    profile('skew', fit_var,
            [stats.skew(x) for x in reduced_intensity],
            prfl_fld, stats_fld, test, plt_fld)
    profile('kurt', fit_var,
            [stats.kurtosis(x) for x in reduced_intensity],
            prfl_fld, stats_fld, test, plt_fld)
    profile('pca', fit_var,
            pca(reduced_intensity),
            prfl_fld, stats_fld, test, plt_fld)
    profile_peakq = np.loadtxt('{0}/profile_peakq.txt'.format(prfl_fld))

    rr = np.array([
                   (x-min(fit_var))/(max(fit_var)-min(fit_var))
                   for x in fit_var
                   ])
    bb = np.array([
                   1-(x-min(fit_var))/(max(fit_var)-min(fit_var))
                   for x in fit_var
                   ])

    for i, _ in enumerate(reduced_intensity):
        # Create intensity plots with q values of interest highlighted
        plt.figure()
        plt.plot(
                 reduced_q,
                 reduced_intensity[i],
                 linestyle='-',
                 color=(rr[i], 0, bb[i]),
                 linewidth=2.0,
                 label='{0:0.1f}{1}'.format(fit_var[i], data_label)
                 )
        plt.axvline(
                    x=profile_peakq[i],
                    linestyle='--',
                    color='C1',
                    label='peakq = {0:0.2f}'.format(profile_peakq[i])
                    )
        plt.legend(loc='upper right')
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.minorticks_on()
        plt.tick_params(which='both', direction='in')
        plt.title(test + ' Curves')
        plt.tight_layout()
        plt.savefig('{0}/curves_{1:0.1f}.png'.format(curves_fld, fit_var[i]))
        plt.close()

    if structure_factor is not None:
        for i, _ in enumerate(structure_factor):
            np.savetxt(
                       '{0}/{1:02d}_{2:0.2f}K.txt'
                       .format(tests_fld, i, temperature[i])
                       .replace('.', 'p'),
                       structure_factor[i]
                       )

            plt.figure()
            plt.plot(
                     reduced_q,
                     structure_factor[i],
                     linestyle='-',
                     color=(rr[i], 0, bb[i]),
                     linewidth=2.0,
                     label='{0:0.1f}{1}'.format(temperature[i], data_label)
                     )
            plt.legend(loc='upper right')
            plt.xlabel('q (Å$^{-1}$)')
            plt.ylabel('Structure Factor (a.u.)')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.minorticks_on()
            plt.tick_params(which='both', direction='in')
            plt.title(test + ' Curves')
            plt.tight_layout()
            plt.savefig('{0}/{1}curves_{2:0.1f}.png'.format(
                                                            sf_fld,
                                                            test,
                                                            temperature[i]
                                                            ))
            plt.close()

    np.savetxt(
               '{0}/{1}/q_range.txt'.format(folder, scan),
               reduced_q
               )
    if pooled:
        np.savetxt(
                   '{0}/{1}/temperature.txt'.format(folder, scan),
                   temperature
                   )
        np.savetxt(
                   '{0}/{1}/positions.txt'.format(folder, scan),
                   y
                   )
    else:
        if temperature is not None:
            np.savetxt(
                       '{0}/{1}/temperature.txt'.format(folder, scan),
                       temperature
                       )
        elif y is not None:
            np.savetxt(
                       '{0}/{1}/positions.txt'.format(folder, scan),
                       y
                       )

    if 'IJ' not in str(scan):
        # Standard deviation plot of all intensities
        plt.figure()
        plt.plot(
                 reduced_q,
                 intensity_std,
                 linewidth='2.0'
                 )
        plt.xlabel('q (Å$^{-1}$)')
        plt.ylabel('SD(Intensity) (a.u.)')
        plt.axvline(
                    x=pinned_q,
                    color='k',
                    linestyle='--'
                    )
        plt.text(
                 pinned_q,
                 0.6*np.mean(intensity_std),
                 'q = {0:02.2f}'.format(pinned_q),
                 horizontalalignment='center',
                 bbox=dict(facecolor='white', alpha=1.0)
                 )
        plt.title('Scan {0}'.format(scan))
        plt.tight_layout()
        plt.savefig('{0}/stdev.png'.format(plt_fld))
        plt.close()

    # Superimposed intensity plot
    plt.figure()
    [
     plt.plot(
              reduced_q,
              x,
              color=(rr[i], 0, bb[i])
              )
     for i, x in enumerate(reduced_intensity)
     ]
    plt.xlabel('q (Å$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.axvline(
                x=pinned_q,
                color='k',
                linestyle='--'
                )
    plt.text(
             pinned_q,
             0.5,
             'q = {0:02.2f}'.format(pinned_q),
             horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=1.0)
             )
    plt.title('Scan {0}'.format(scan))
    plt.tight_layout()
    plt.savefig('{0}/superimposedcurves.png'.format(plt_fld))
    plt.close()

    # Save the calibration data sets and log the date/time processing was done
    if temperature is not None and ramping is False and 'IJ' not in str(scan):
        with open('{0}/{1}/{1}_data.pckl'.format(folder, scan), 'wb') as f:
            pickle.dump([temperature, reduced_q, reduced_intensity], f)
        with open('{0}/{1}/{1}_log.txt'.format(folder, scan), 'a+') as f:
            f.write(datetime.now().strftime("\n%d-%b-%Y %I:%M:%S %p"))

    # Save the ethanol (cold/ambient/hot) and water impinging jet data sets
    # and log the date/time processing was done
    elif y is not None and ramping is False:
        with open('{0}/{1}/{1}_data.pckl'.format(folder, scan), 'wb') as f:
            pickle.dump([y, reduced_q, reduced_intensity], f)
        with open('{0}/{1}/{1}_log.txt'.format(folder, scan), 'a+') as f:
            f.write(datetime.now().strftime("\n%d-%b-%Y %I:%M:%S %p"))

    # Save the ethanol ramping impinging jet data set and log the date/time
    # processing was done
    elif ramping is True:
        with open(folder + '/' + str(scan) + '/' + str(scan).rsplit('/')[-1] +
                  '_data.pckl', 'wb') as f:
            pickle.dump([temperature, y, reduced_q, reduced_intensity], f)
        with open(folder + '/' + str(scan) + '/' + str(scan).rsplit('/')[-1] +
                  '_log.txt', 'a+') as f:
            f.write(datetime.now().strftime("\n%d-%b-%Y %I:%M:%S %p"))
