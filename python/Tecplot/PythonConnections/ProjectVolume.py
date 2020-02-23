# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:52:06 2019

@author: rahmann
"""

import numpy as np
import tecplot as tp
from tecplot.constant import SliceSource, SliceSurface
from tecplot.data.operate import execute_equation

tp.session.connect()

def main():
    (x, y, z) = tp.active_frame().dataset.zone(0).dimensions
    
    tp.macro.execute_command('$!DRAWGRAPHICS FALSE')
    # X projection
    for i in range(x):
        tp.active_frame().plot().slice(0).origin=(i,0,0)
        tp.macro.execute_command('''$!ExtractSlices 
                                 Group = 1
                                 ExtractMode = SingleZone''')
        if i > 0:
            execute_equation('''{Intensity} = {Intensity}[2] + {Intensity}[3]''', zones=tp.active_frame().dataset.zone(1))
            tp.macro.execute_command('$!DeleteZones  [3]')
    tp.macro.execute_command('$!DRAWGRAPHICS TRUE')
    tp.macro.execute_command('$!RedrawAll')
    
with tp.session.suspend():
    main()