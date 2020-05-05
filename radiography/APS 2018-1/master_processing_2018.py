"""
-*- coding: utf-8 -*-
Runs all the major scripts for full processing of the APS 2018-1 data set.
@Author: rahmann
@Date:   2020-04-22 21:14:21
@Last Modified by:   rahmann
@Last Modified time: 2020-04-22 21:14:21
"""

import time
from datetime import timedelta
from whitebeam_2018 import main as whitebeam_2018
from energy_spectra_2018_plots import main as energy_spectra_2018_plots
from jet_processing import main as jet_processing
from jet_analysis import main as jet_analysis
from jet_correction import main as jet_correction
from jet_errors import run_main as jet_errors

# Build the whitebeam models
start = time.time()
whitebeam_2018()
wb_time = timedelta(seconds=(time.time() - start))
print('1/6: whitebeam_2018 finished ({0}).'.format(wb_time))

# Create the energy spectra plots
start = time.time()
energy_spectra_2018_plots()
es_time = timedelta(seconds=(time.time() - start))
print('2/6: energy_spectra_2018_plots finished ({0}).'.format(es_time))

# Process the uniform jets
start = time.time()
jet_processing()
jp_time = timedelta(seconds=(time.time() - start))
print('3/6: jet_processing finished ({0})'.format(jp_time))

# Analyze the uniform jets
start = time.time()
jet_analysis()
ja_time = timedelta(seconds=(time.time() - start))
print('4/6: jet_analysis finished ({0}).'.format(ja_time))

# Apply the correction factors (from jet_analysis) to the uniform jets
start = time.time()
jet_correction()
jc_time = timedelta(seconds=(time.time() - start))
print('5/6: jet_correction finished ({0}).'.format(jc_time))
#Apply the correction factors (from jet_analysis) to the uniform jets
start = time.time()
jet_errors()
je_time = timedelta(seconds=(time.time() - start))
print('6/6: jet_errors finished ({0}).'.format(je_time))
