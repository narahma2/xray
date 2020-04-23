"""
-*- coding: utf-8 -*-
Runs all the major scripts for full processing of the APS 2018-1 data set.
@Author: rahmann
@Date:   2020-04-22 21:14:21
@Last Modified by:   rahmann
@Last Modified time: 2020-04-22 21:14:21
"""

from whitebeam_2018 import main as whitebeam_2018
from jet_processing import main as jet_processing

# Build the whitebeam models
whitebeam_2018()

# Process the uniform jets
jet_processing()