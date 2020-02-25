# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:40:57 2019

@author: rahmann
"""

import tecplot as tp
import numpy as np
from tecplot.constant import ExportRegion, Projection
import glob
from timeit import default_timer as timer
from pathlib import Path

# MAKE SURE TO EDIT LINES 20, 22, 25 (OPTIONAL: LINES 45 - 54).
# IMAGES ARE EXPORTED WITH A WIDTH OF 960, IF YOU WANT TO CHANGE THEN EDIT 'width=960' ON LINE 64

start = timer()

test = Path('FOLDER CONTAINING THE .PLT FILES')
inputSet = glob.glob(str(test) / '*.plt')
style = Path('PATH TO STYLES .STY FILE')
frame = tp.active_frame()

rotation_angle = 'ANGLE TO ROTATE BY' # e.g. replace string with 18 (without quotation marks) to get 360/18 = 20 rotated views
angles = np.linspace(start=0, stop=360-rotation_angle, num=360/rotation_angle)

# First file inputSet[0] is read in, change 0 to any index needed within inputSet
# Only one volume is rotated and exported in this script, change this around if you want to do a time sequence
## similar to how the Volumes_TimeSequence.py script is setup
dataSet = tp.data.load_tecplot(inputSet[0], read_data_option=2)
zone = dataSet.zone(0)
frame.plot().fieldmap(0).surfaces.surfaces_to_plot = None
tp.macro.execute_command("""$!ReadStyleSheet  """ + str(style) + """
                         IncludePlotStyle = Yes
                         IncludeText = Yes
                         IncludeGeom = Yes
                         IncludeAuxData = Yes
                         IncludeStreamPositions = Yes
                         IncludeContourLevels = Yes
                         Merge = No
                         IncludeFrameSizeAndPosition = No""")

# If you want to change the intensities of the isocontours shown
change_colors = 'False'
if change_colors:
    # Three isocontours shown, match your style sheet for the number shown
    frame.plot().isosurface(0).isosurface_values[0]=0.02
    frame.plot().isosurface(1).isosurface_values[0]=0.04
    frame.plot().isosurface(2).isosurface_values[0]=0.06
    # Contour levels shown (on the legend), match your style sheet as needed
    frame.plot().contour(0).levels.reset_levels([0.02, 0.04, 0.06])
    frame.plot().contour(0).colormap_filter.continuous_min=0.02
    frame.plot().contour(0).colormap_filter.continuous_max=0.06

tp.active_frame().plot().view.projection=Projection.Perspective
tp.macro.execute_command('$!FrameLayout ShowBorder = No')

Path(test / 'Export' / style.name.split('.')[0] / 'Rotation').mkdir(parents=True, exist_ok=True)
frame.active_zones(0)

for i in angles:
    tp.export.save_tiff(str(test / 'Export' / style.name.split('.')[0] / 'Rotation') + '/View' + f"{round(i):03d}" + '.tiff',
                        width=960,
                        region=ExportRegion.CurrentFrame,
                        supersample=3)
    tp.macro.execute_command('''$!Rotate3DView Y
                             Angle = ''' + str(i) + '''
                             RotateOriginLocation = DefinedOrigin''')
    
elapsed_time = timer() - start
f = open(str(test / 'Export' / style.name.split('.')[0] / 'Rotation') + '/log.txt', 'w' )
f.write( 'Time elapsed = ' + str(elapsed_time/60) + ' minutes\n' )
f.close()