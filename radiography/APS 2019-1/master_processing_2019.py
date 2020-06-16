"""
-*- coding: utf-8 -*-
Runs all the major scripts for full processing of the APS 2019-1 data set.
@Author: rahmann
@Date:   2020-05-29
@Last Modified by:   rahmann
"""

import time
from datetime import timedelta
from whitebeam_2019 import main as whitebeam_2019
from calibration_processing import main as cal_processing
from calibration_analysis import main as cal_analysis
from calibration_correction import main as cal_correction
from calibration_errors import main as cal_errors
from spray_processing import main as spr_processing
from watervsKI import main as watervsKI
from monobeam_radiography import main as mbr
from monobeam_radiography_vertical import main as mbr_vert

# Build the whitebeam models
# Output: Data under /Model
start = time.time()
whitebeam_2019()
wb_time = timedelta(seconds=(time.time() - start))
print('1/9: whitebeam_2019 finished ({0}).'.format(wb_time))

# Process the calibrations
# Output: Data under /Processing folder
start = time.time()
cal_processing()
cp_time = timedelta(seconds=(time.time() - start))
print('2/9: cal_processing finished ({0})'.format(cp_time))

# Analyze the calibrations
# Output: CF files under /Processing folder; Plots under /Figures/Cal_Summary
start = time.time()
cal_analysis()
ca_time = timedelta(seconds=(time.time() - start))
print('3/9: cal_analysis finished ({0}).'.format(ca_time))

# Apply the correction factors (from cal_analysis) to the calibrations
# Output: Data under /Corrected folder
start = time.time()
cal_correction()
cc_time = timedelta(seconds=(time.time() - start))
print('4/9: cal_correction finished ({0}).'.format(cc_time))

# Find errors in the final corrected jets
# Output: Plots under /Figures/Cal_Errors
start = time.time()
cal_errors()
ce_time = timedelta(seconds=(time.time() - start))
print('5/9: cal_errors finished ({0}).'.format(ce_time))

# Process the LCSC spray
# Output: Plots under /Images/Spray/EPL
start = time.time()
spr_processing()
sp_time = timedelta(seconds=(time.time() - start))
print('6/9: spray_processing finished ({0}).'.format(sp_time))

# Compare the water and KI LCSC spray at similar conditions
# Output: Plots under /WatervsKI
start = time.time()
watervsKI()
wk_time = timedelta(seconds=(time.time() - start))
print('7/9: watervsKI finished ({0}).'.format(wk_time))

# Compare the water WB and MB scans (horizontal) at various conditions
# Output: Plots under /Monobeam
start = time.time()
mbr()
mh_time = timedelta(seconds=(time.time() - start))
print('8/9: monobeam_radiography finished ({0}).'.format(mh_time))

# Compare the water WB and MB scans (vertical) at various conditions
# Output: Plots under /Monobeam
start = time.time()
mbr()
mv_time = timedelta(seconds=(time.time() - start))
print('9/9: monobeam_radiography_vertical finished ({0}).'.format(mv_time))
