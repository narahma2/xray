# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:59:04 2018

@author: Naveed
"""

import os
import time
import numpy as np
from PIL import Image

def stackAvg(folder, stopwatch=0):
    start = time.time()
    data = 0    # array to hold the image files
    count = 0   # number of files stored in data
    
    for filename in os.listdir(folder):
        # Get only .tif/.tiff files within specified folder
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            im = Image.open(folder + '/' + filename)    # open as PIL.Image object
            data += np.array(im)                        # build data array
            count += 1                                  # number of files opened
    output = data / count     # averaged image                                                 
    
    end = time.time()
    if stopwatch:
        # Display processing time if requested
        print(end - start)
        
    return output

def darkSub(I, dark):
    I_im = np.array(Image.open(I))
    dark_im = np.array(Image.open(dark))
    output = I_im - dark_im
    
    return output

def normalize(I, I_0, dark):
    I_im = np.array(Image.open(I))
    I_0_im = np.array(Image.open(I_0))
    dark_im = np.array(Image.open(dark))
    output = (I_im - dark_im) / (I_0_im - dark_im)
    
    return output