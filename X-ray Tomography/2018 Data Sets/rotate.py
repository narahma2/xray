# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:31:39 2019

@author: rahmann
"""

import tecplot as tp
#from tecplot.exception import *
#from tecplot.constant import *
import tecplot as tp
from tecplot.constant import ExportRegion, Projection
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

condition = ['IJ_Mixing']
flowrate = [45]
styles = ['volume_XY']

for cond in condition:
    for style in styles:
        for k in flowrate:
            start = timer()
            test = 'R:/XrayTomo/2018_DataSets/' + cond + '/Data/' + cond + '_0p' + str(k) + 'gpm/LargerGrid/7Passes_5x5_Smoothing'
            inputSet = glob.glob(test + '/PLT/*.plt')
            frame = tp.active_frame()
              
            dataSet = tp.data.load_tecplot(inputSet[0], read_data_option=2)
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
            
            pathlib.Path(test + '/Export/0p' + str(k) + 'gpm_rotation').mkdir(parents=True, exist_ok=True)
            frame.active_zones(0)
            tp.macro.execute_command("""$!RotateData ZoneList =  [""" + str(1) + """] Angle = -10 XVar = 1 YVar = 2 ZVar = 3 NormalX = 0 NormalY = 1 NormalZ = 0""")
            
            angles = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342]
            tp.active_frame().plot().view.projection=Projection.Perspective
            tp.macro.execute_command('$!FrameLayout ShowBorder = No')
            
            for i in angles:
                tp.export.save_tiff(test + '/Export/0p' + str(k) + 'gpm_rotation/Vol1_View' + str(i) + '.tiff',
                                    width=1049,
                                    region=ExportRegion.CurrentFrame,
                                    supersample=5)
                tp.macro.execute_command('''$!Rotate3DView Y
                                         Angle = 18
                                         RotateOriginLocation = DefinedOrigin''')
                
            elapsed_time = timer() - start
            f = open( test + '/log.txt', 'w' )
            f.write( 'Time elapsed = ' + str(elapsed_time/60) + ' minutes\n' )
            f.close()