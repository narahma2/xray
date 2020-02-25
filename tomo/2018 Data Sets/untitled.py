import tecplot as tp
from tecplot.exception import *
from tecplot.constant import *

# Uncomment the following line to connect to a running instance of Tecplot 360:
# tp.session.connect()

tp.macro.execute_command('$!RedrawAll')
tp.macro.execute_command('''$!RotateData 
  ZoneList =  [1]
  Angle = -90
  XVar = 1
  YVar = 2
  ZVar = 3
  OriginX = 3.98
  OriginY = -0.48
  OriginZ = -0.53
  NormalX = 1
  NormalY = 0
  NormalZ = 0''')
tp.macro.execute_command('$!RedrawAll')
tp.macro.execute_command('''$!RotateData 
  ZoneList =  [1]
  Angle = -180
  XVar = 1
  YVar = 2
  ZVar = 3
  OriginX = 3.98
  OriginY = -0.48
  OriginZ = -0.53
  NormalX = 1
  NormalY = 0
  NormalZ = 0''')
tp.macro.execute_command('$!RedrawAll')
# End Macro.

