import tecplot as tp
from tecplot.exception import *
from tecplot.constant import *

# Uncomment the following line to connect to a running instance of Tecplot 360:
# tp.session.connect()

tp.data.operate.execute_equation(equation='{X_Proj} = {Intensity}',
    j_range=tp.data.operate.Range(max=0),
    k_range=tp.data.operate.Range(max=0))
# End Macro.

