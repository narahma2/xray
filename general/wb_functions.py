# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:26:03 2019

@author: rahmann
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import sg_filt, find_peaks, peak_widths


def convert2EPL(test_path, offset, model_pckl, cm_pix,
                dark_path, flat_path, cropped_view=None, plot=False):
    """
    Function that takes in the designated raw test and converts to EPL.
    Optionally plots the result as well.
    =============
    --VARIABLES--
    test_path:          Path to an individual raw test image (.tif)
    offset:             Background EPL value correction. Given as either a
                        tuple of slices that designate a region outside the
                        spray, or as a float if the offset value is known
                        beforehand.
    model_pckl:         White beam model to be used. Generated by the
                        specific whitebeam_*.py script (.pckl).
    cm_pix:             Pixel to cm conversion, dependent on experimental
                        setup (scalar).
    dark_path:          Path to the image to be used for dark current
                        subtraction (.tif).
    flat_path:          Path to the image to be used for flat field
                        normalization (.tif).
    cropped_view:       Optional integer array parameter to crop down the
                        image before conversion. Defaulted to None (int).
    plot:               Check to see if plots are wanted. Defaulted to
                        False (boolean).
    """

    f = open(model_pckl, 'rb')
    model = pickle.load(f)
    f.close()

    dark = np.array(Image.open(dark_path))
    flatfield = np.array(Image.open(flat_path))
    flatfield_darksub = flatfield - dark

    beam_middle = np.zeros((flatfield_darksub.shape[1]))
    for i in range(flatfield_darksub.shape[1]):
        beam_middle[i] = np.argmax(
                                   sg_filt(
                                                 flatfield_darksub[:, i],
                                                 55,
                                                 3
                                                 )
                                   )

    beam_middle_avg = int(np.mean(beam_middle).round())

    angles = model[0]
    angle_to_px = [
                   beam_middle_avg+3500*np.tan(x*10**-3)/cm_pix
                   for x in angles
                   ]

    data = np.array(Image.open(test_path))
    data_norm = (data-dark) / flatfield_darksub

    data_epl = np.empty(np.shape(data_norm), dtype=float)

    if isinstance(cropped_view, None):
        cropped_view = np.linspace(1, data_norm.shape[0])

    for z, _ in enumerate(cropped_view):
        j = np.argmin(abs(z-np.array(angle_to_px)))
        data_epl[z, :] = model[1][j](data_norm[z, :])

    if type(offset) == tuple:
        offset_epl = data_epl[offset[0], offset[1]]
        data_epl -= offset_epl
    else:
        data_epl -= offset

    if plot:
        plt.imshow(data_epl, vmin=0, vmax=1)
        plt.colorbar
        plt.title('EPL [cm] Mapping of Spray')

    return data_epl


def ellipse(data_epl, pos, cm_px, peak_width=20, rel_height=0.8,
            plot=False):
    """
    Function that takes in the EPL image and finds the best elliptical fit
    EPL diameter. Optionally plots the result against the 'optical diameter'
    as well for quick diagnostic of single image.
    =============
    --VARIABLES--
    data_epl:           EPL converted image of spray from convert2EPL
                        function (array).
    pos:                Vertical pixel location for the elliptical fitting
                        plot (integer).
    peak_width:         Width in pixels to be used for the find_peaks
                        function. Defaulted to 20 px (integer).
    rel_height:         Relative height to be used for the peak_widths
                        function. Defaulted to 0.8 for the 'fifth_maximum'
                        (scalar).
    cm_px:             Pixel to cm conversion, dependent on experimental
                        setup (scalar).
    plot:               Check to see if plots are wanted. Defaulted to
                        False (boolean).
    """

    peaks, _ = find_peaks(
                          sg_filt(data_epl[pos, :], 105, 7),
                          width=peak_width,
                          prominence=0.1
                          )
    [rel_width, rel_max, lpos, rpos] = peak_widths(
                                                   sg_filt(
                                                           data_epl[pos, :],
                                                           105,
                                                           7
                                                           ),
                                                   peaks,
                                                   rel_height=rel_height
                                                   )
    rel_width = rel_width[0]
    rel_max = rel_max[0]
    lft = int(round(lpos[0]))
    rgt = int(round(rpos[0]))
    model_epl, fitted_graph, epl_graph = ideal_ellipse(
                                                       data_epl[pos, :]
                                                               [lft:rgt],
                                                       rel_width,
                                                       rel_max,
                                                       cm_px
                                                       )

    if plot:
        plot_ellipse(epl_graph, fitted_graph)

        plt.figure()
        plt.imshow(data_epl, vmin=0, vmax=1)
        plt.plot([pos]*len(data_epl[pos, :]), color='r', linestyle='--')
        plt.colorbar()
        plt.title('EPL [cm] Mapping of Spray')

        plt.figure()
        plt.plot(data_epl[pos, :])
        plt.title('Line Scan at ' + str(pos))
        plt.xlabel('Horizontal Location [px]')
        plt.ylabel('EPL [cm]')


def ideal_ellipse(y, rel_width, rel_max, dx, units='cm'):
    """
    Function that takes in a line scan and returns the best fit ellipse.
    =============
    --VARIABLES--
    y:                  Intensity line scan values (array).
    rel_width:          Width of peak.
    rel_max:            Y value corresponding to the peak width (float).
    dx:                 Pixel size for x dimension (float).
    units:              Units of the x dimension (string: 'cm', 'mm').
    """

    x = np.linspace(start=-(len(y)*dx)/2, stop=(len(y)*dx)/2, num=len(y))
    y = y - rel_max

    area = np.trapz(y, dx=dx)
    a = rel_width/2*dx
    fitted_radius = 2*area / (np.pi*a)
#    if units == 'cm':
#        b = np.linspace(0,1,1000)
#    elif units == 'mm':
#        b = np.linspace(0,10,1000)
#    a = relative_width/2*dx
#    minimize = np.zeros(len(b))
#
#    for i, R in enumerate(b):
#        check_area = (1/2)*np.pi*a*R
#        minimize[i] = abs(area - check_area)
#
    y = y + rel_max
#
#    fitted_radius = b[np.argmin(minimize)]
    data_graph = {'x': x, 'y': y}
    fitted_graph = {'center': (0, rel_max), 'a': a, 'b': fitted_radius}

    return fitted_radius, fitted_graph, data_graph


def plot_ellipse(data_graph, fitted_graph, save_path=None):
    t = np.linspace(0, np.pi)
    a = fitted_graph['a']
    b = fitted_graph['b']
    xc = fitted_graph['center'][0]
    yc = fitted_graph['center'][1]

    plt.figure()
    plt.plot(
             10*data_graph['x'],
             10*data_graph['y'],
             label='Data w/ Full Width = {0:0.2f} mm'.format(10*a*2)
             )
    plt.plot(
             10*(xc+a*np.cos(t)),
             10*(yc+b*np.sin(t)),
             label='Fitted Ellipse w/ Diameter = {0:0.2f} mm'.format(10*b)
             )
    plt.legend()
    plt.title('Ellipse Fitting to EPL Scan')
    plt.xlabel('Horizontal Location (mm)')
    plt.ylabel('EPL (mm)')
    if save_path is not None:
        plt.savefig(save_path)


def plot_widths(ydata, peaks, relative_max, lpos, rpos, save_path):
    plt.figure()
    plt.plot(ydata, color='blue')
    plt.plot(peaks, ydata[peaks], 'x', color='orange')
    plt.hlines(relative_max, xmin=lpos, xmax=rpos, color='red')
    plt.title('EPL Scan Widths, Peak = {0:0.2f}'.format(ydata[peaks[0]]))
    plt.ylabel('EPL (cm)')
    if save_path is not None:
        plt.savefig(save_path)
