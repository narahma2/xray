function T = tube_model(window, window_epl)
%TUBE_MODEL         Create tube model for the calibration data.
%
% Inputs:
%           window:         Window material from NIST_XCOM.xlsx file.
%           window_epl:     Thickness of window (in mm).
%
% Outputs:
%           T:              Expected transmission value (I/I0).
%
% Author:           Naveed Rahman
% Creation Date:    5 June 2020

clc, clear, close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load inputs and pre-process source spectra
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set working directory
cd 'path/to/directory'

% Load in SpekCalc spectra
[sc_I, sc_keV] = spekcalc(['path/to/script/80kVp'])

% Load attenuation coefficient curves
% THESE ATTENUATION COEFFICIENTS ARE IN UNITS OF 1/MM
% NO NEED TO MULTIPLY BY DENSITY!
[att_keV, att_coeff] = nist_xcom({'Air', 'Be', 'CsI', window}, 8, 'mm', 80);

% Re-shape SpekCalc spectra to match the XCOM keV
I0 = interp1(sc_keV, sc_I, att_keV);
keV = att_keV;

% Remove NaN
I0(isnan(I0)) = 0;

% Convert y-axis intensity to # of visible photons (multiply by keV)
% Logic being that a 10 keV x-ray photon will emit 10x more visible photons
%   than a 1 keV photon.
I0 = I0 .* keV;

% Re-shape the XCOM spectra to match the final source spectra
att_coeff = xcom_reshape(att_keV, att_coeff, keV);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Experimental setup (path lengths/thicknesses)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EPL setup (mm)
air_epl = 500;
be_epl = 1;
CsI_epl = 0.150;    % CsI w/ FOS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Scintillator response & Beer-Lambert law
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scintillator response
scint_tran = beer_lambert(I0, att_coeff.CsI, CsI_epl);
scint_detected = I0 - scint_tran;
scint_response = scint_detected ./ I0;

% Remove NaN
scint_response(isnan(scint_response)) = 0;

% Filter the spectra through air/Be window
filtered_spectra = beer_lambert(I0, att_coeff.Air, air_epl);
filtered_spectra = beer_lambert(filtered_spectra, att_coeff.Be, be_epl);
detected_spectra_I0 = filtered_spectra .* scint_response

% Apply the window material and path length
spectra_I = beer_lambert(filtered_spectra, att_coeff.(window), window_epl);

% Find detected spectra
detected_spectra_I = spectra_I .* scint_response;

% Calculate expected transmission (T) value
T = trapz(keV, detected_spectra_I) / trapz(keV, detected_spectra_I0);