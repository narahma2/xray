# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:30:00 2019

@author: rahmann
"""

import sys
sys.path.insert(0, "E:/OneDrive - purdue.edu/Research/GitHub/coding/python/Tecplot")

import pathlib
import tecplot as tp
import matplotlib.pyplot as plt
from tecplot.data.operate import execute_equation
from tecplot.constant import PlotType
from timeit import default_timer as timer
from d10_to_tecplot import convert_tp, save_plt, volume_masking, volume_smoothing

im7_path = "E:\\DaVisProjects\\DaVis10_IJ_Sept2019\\0p30\\AddCameraAttributes\\7Passes_5x5 GS\\Mask_01\\ConvertToMM" + \
            "\\MART_(iterations=10,smooth=0.5,10,sparse=0)\\B00001.im7"

tp.session.connect()

with tp.session.suspend():
    data = tp.active_frame().create_dataset('Data', ['X', 'Y', 'Z', 'Intensity'], reset_style=True)
    convert_tp(im7_path, dataset=data, zone_name='EPL (mm)', crop_x1=-1,crop_x2=9,crop_y1=-12,crop_y2=11,crop_z1=-11,crop_z2=10)
    
    tp.macro.execute_command('''$!RotateData ZoneList =  [1] Angle = 90 
                             XVar = 1 YVar = 2 ZVar = 3
                             OriginX = 3.98 OriginY = -0.48 OriginZ = -0.53
                             NormalX = 1 NormalY = 0 NormalZ = 0''')
    
    tp.macro.execute_command('''$!RotateData ZoneList =  [1] Angle = -11 
                             XVar = 1 YVar = 2 ZVar = 3
                             OriginX = 3.98 OriginY = -0.48 OriginZ = -0.53
                             NormalX = 0 NormalY = 0 NormalZ = 1''')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    