# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:56:05 2018

@author: Naveed
"""

import numpy as np

def openXCOM(material, c):
    data = np.loadtxt(material)
    
#   XCrossSec XOP
#   c=0 keV
#   c=1 Rayleigh
#   c=2 Compton
#   c=3 Photoelectic Absorption
#   c=6 Total
#   c=7 Total - Ray
    
    return data[:,c]

def norm2EPL(p, normalized):
    # p is the polynomial fit coefficients for the overall system attenuation coefficient
    # normalized is the normalized X-ray radiograph
    epl = -np.log(normalized) / np.polyval(p, normalized)
    
    return epl