"""
Created 28 May 2018 16:56:00 EST

@author: rahmann
"""

import sys
sys.path.append("..")

import os
import numpy as np
from scipy import stats
from Form_Factor.xray_factor import load_qspace
from matplotlib import pyplot as plt
from Statistics.CIs_LinearRegression import run_all as lin_fit


folder = "D:/Naveed/2018Feb APS/Temperature/DAWN"

rampup = [load_qspace(folder + "/RampUp/" + filename) for filename in os.listdir(folder + "/RampUp")]
rampup_temperature = np.array([2.5, 11, 20, 29, 39, 49, 60, 70, 87])

rampdown = [load_qspace(folder + "/RampDown/" + filename) for filename in os.listdir(folder + "/RampDown")]
rampdown_temperature = np.array([87, 80, 69, 59, 48, 40, 31, 21, 11, 4])

rampup_peak = [stats.tmax(rampup[i]["S"]) for i in range(len(rampup))]
rampup_skew = [stats.skew(rampup[i]["S"]) for i in range(len(rampup))]
rampup_kurt = [stats.kurtosis(rampup[i]["S"]) for i in range(len(rampup))]

uppeak_fitobj, uppeak_yerr, uppeak_x, uppeak_y = lin_fit(rampup_temperature, rampup_peak, xlabel="Temperature [C]", \
                                                         title="Ramp Up - Peak")
upskew_fitobj, upskew_yerr, upskew_x, upskew_y = lin_fit(rampup_temperature, rampup_skew, xlabel="Temperature [C]", \
                                                         title="Ramp Up - Skewness")
upkurt_fitobj, upkurt_yerr, upkurt_x, upkurt_y = lin_fit(rampup_temperature, rampup_kurt, xlabel="Temperature [C]", \
                                                         title="Ramp Up - Kurtosis")

rampdown_peak = [stats.tmax(rampdown[i]["S"]) for i in range(len(rampdown))]
rampdown_skew = [stats.skew(rampdown[i]["S"]) for i in range(len(rampdown))]
rampdown_kurt = [stats.kurtosis(rampdown[i]["S"]) for i in range(len(rampdown))]

downpeak_fitobj, downpeak_yerr, downpeak_x, downpeak_y = lin_fit(rampdown_temperature, rampdown_peak, xlabel="Temperature [C]", \
                                                         title="Ramp Down - Peak")
downskew_fitobj, downskew_yerr, downskew_x, downskew_y = lin_fit(rampdown_temperature, rampdown_skew, xlabel="Temperature [C]", \
                                                         title="Ramp Down - Skewness")
downkurt_fitobj, downkurt_yerr, downkurt_x, downkurt_y = lin_fit(rampdown_temperature, rampdown_kurt, xlabel="Temperature [C]", \
                                                         title="Ramp Down - Kurtosis")

plt.figure()
plt.plot(rampup[0]["q"], rampup[0]["S"], label="2.5C")
plt.plot(rampup[1]["q"], rampup[1]["S"], label="11C")
plt.plot(rampup[2]["q"], rampup[2]["S"], label="20C")
plt.plot(rampup[3]["q"], rampup[3]["S"], label="29C")
plt.plot(rampup[4]["q"], rampup[4]["S"], label="39C")
plt.plot(rampup[5]["q"], rampup[5]["S"], label="49C")
plt.plot(rampup[6]["q"], rampup[6]["S"], label="60C")
plt.plot(rampup[6]["q"], rampup[6]["S"], label="70C")
plt.plot(rampup[6]["q"], rampup[6]["S"], label="87C")
plt.legend()

plt.figure()
plt.plot(rampdown[0]["q"], rampdown[0]["S"], label="87C")
plt.plot(rampdown[1]["q"], rampdown[1]["S"], label="80C")
plt.plot(rampdown[2]["q"], rampdown[2]["S"], label="69C")
plt.plot(rampdown[3]["q"], rampdown[3]["S"], label="59C")
plt.plot(rampdown[4]["q"], rampdown[4]["S"], label="48C")
plt.plot(rampdown[5]["q"], rampdown[5]["S"], label="40C")
plt.plot(rampdown[6]["q"], rampdown[6]["S"], label="31C")
plt.plot(rampdown[6]["q"], rampdown[6]["S"], label="21C")
plt.plot(rampdown[6]["q"], rampdown[6]["S"], label="11C")
plt.plot(rampdown[7]["q"], rampdown[7]["S"], label="4C")
plt.legend()
