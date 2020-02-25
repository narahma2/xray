import tecplot as tp
from tecplot.exception import *
from tecplot.constant import *

# Uncomment the following line to connect to a running instance of Tecplot 360:
# tp.session.connect()

tp.active_frame().plot().slice(0).orientation=SliceSurface.XPlanes
tp.active_frame().plot().slice(0).orientation=SliceSurface.IPlanes
tp.active_frame().plot().slice(0).origin=(2,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(6,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(11,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(15,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(18,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(19,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(15,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(11,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(5,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.active_frame().plot().slice(0).origin=(0,
    tp.active_frame().plot().slice(0).origin[1],
    tp.active_frame().plot().slice(0).origin[2])
tp.macro.execute_command('''$!ExtractSlices 
  Group = 1
  ExtractMode = SingleZone''')
# End Macro.

