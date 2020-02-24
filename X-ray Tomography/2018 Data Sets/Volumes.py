# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:20:48 2019

@author: rahmann
"""

import tecplot as tp
#from tecplot.exception import *
#from tecplot.constant import *
import tecplot as tp
from tecplot.constant import ExportRegion
import logging
import os
from os import listdir
from os.path import isfile, join
import glob
from tkinter import filedialog
from tkinter import Tk
from pathlib import Path
from timeit import default_timer as timer
import pathlib 

start = timer()

condition = ['IJ', 'IJ_Mixing']
flowrate = [30, 45]
styles = ['volume_iso', 'volume_XY', 'volume_XZ']

for cond in condition:
    for style in styles:
        for k in flowrate:
            start = timer()
            test = 'R:/XrayTomo/2018_DataSets/' + cond + '/Data/' + cond + '_0p' + str(k) + 'gpm/LargerGrid/7Passes_5x5_Smoothing'
            inputSet = glob.glob(test + '/PLT/*.plt')
            frame = tp.active_frame()
            
            i = -1
            for jj in inputSet:
                i += 1
                dataSet = tp.data.load_tecplot(jj, read_data_option=2)
                zone = dataSet.zone(0)
                frame.plot().fieldmap(0).surfaces.surfaces_to_plot = None
                tp.macro.execute_command("""$!ReadStyleSheet  "E:/XrayTomo/2018_DataSets/IJ/Styles/""" + style + """.sty"
                  IncludePlotStyle = Yes
                  IncludeText = Yes
                  IncludeGeom = Yes
                  IncludeAuxData = Yes
                  IncludeStreamPositions = Yes
                  IncludeContourLevels = Yes
                  Merge = No
                  IncludeFrameSizeAndPosition = No""")
                
                if cond == 'IJ_Mixing':
                    frame.plot().isosurface(0).isosurface_values[0]=0.02
                    frame.plot().isosurface(1).isosurface_values[0]=0.04
                    frame.plot().isosurface(2).isosurface_values[0]=0.06
                    frame.plot().contour(0).levels.reset_levels([0.02, 0.04, 0.06])
                    frame.plot().contour(0).colormap_filter.continuous_min=0.02
                    frame.plot().contour(0).colormap_filter.continuous_max=0.06
                
                pathlib.Path(test + '/Export/0p' + str(k) + 'gpm_' + style).mkdir(parents=True, exist_ok=True)
                frame.active_zones(0)
                tp.macro.execute_command("""$!RotateData ZoneList =  [""" + str(1) + """] Angle = -10 XVar = 1 YVar = 2 ZVar = 3 NormalX = 0 NormalY = 1 NormalZ = 0""")
                tp.export.save_tiff(test + '/Export/0p' + str(k) + 'gpm_' + style + '/' + f"{i:02d}" + '.tiff', width=462, region=ExportRegion.CurrentFrame, supersample=6)
                
            elapsed_time = timer() - start
            f = open( test + '/Export/0p' + str(k) + 'gpm_' + style + '/log.txt', 'w' )
            f.write( 'Time elapsed = ' + str(elapsed_time/60) + ' minutes\n' )
            f.close()