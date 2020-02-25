# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:22:59 2019

@author: rahmann
"""

import glob
import numpy as np
from os import path
from libim7 import readim7

im7_folder = "E:\\DaVisProjects\\DaVis10_IJ_Sept2019\\IJ\\0p30\\5Passes_5x5 Gaussian smoothing filter\\MaskOutImage\\" + \
                   "AddCameraAttributes\\AddAttributes\\CompressExpand\\Mult\\MART_(iterations=1000,smooth=0.5,1000,sparse=0)\\Data\\S00001\\"
                   
im7_files = glob.glob(im7_folder + "*.im7")

for n, im7_file in enumerate(im7_files):
    buf, att = readim7(im7_file)
    
    if n == 0:
        # Number of voxels in X, Y, Z
        nx = buf.nx
        ny = buf.ny
        nz = len(im7_files)
        volume = np.zeros((nx, ny, nz))
        
    volume[:,:,n] = buf.blocks.T[:,:,0]
    
# Millimeters per voxel from calibration
scale_x = float(att['_SCALE_X'].rsplit(' ')[0])
scale_y = float(att['_SCALE_Y'].rsplit(' ')[0])
scale_z = float(att['_SCALE_Z'].rsplit(' ')[0])

# Starting point in X, Y, Z in millimeters
start_x = float(att['_SCALE_X'].rsplit('\nmm')[0].rsplit(' ')[1])
start_y = float(att['_SCALE_Y'].rsplit('\nmm')[0].rsplit(' ')[1])
start_z = float(att['_SCALE_Z'].rsplit('\nmm')[0].rsplit(' ')[1])

# Create volume grid extents in X, Y, Z based on calibrated millimeter dimensions
x_mm = np.linspace(start=start_x, stop=start_x + nx*scale_x, num=nx, dtype=np.float32)
y_mm = np.linspace(start=start_y, stop=start_y + ny*scale_y, num=ny, dtype=np.float32)
z_mm = np.linspace(start=start_z, stop=start_z + nz*scale_z, num=nz, dtype=np.float32)