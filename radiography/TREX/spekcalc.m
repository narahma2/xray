function [intensity, keV] = spekaclc(spekcalc_file)
%SPEKCALC       Reads in SpekCalc spectra.
%
% Syntax:
%           [intensity, keV] = spekcalc(spekcalc_file)
%
% Input:
%           spekcalc_file:      Path to *.spec file.
%
% Output:
%           intensity:          Spectra intensity.
%           keV:                Photon energy in keV.
%
% Author:           Naveed Rahman
% Creation Date:    5 June 2020

f = fopen(spekcalc_file)
data_spekcalc = textscan(f, '%f %f', 'HeaderLines', 18);
fclose(f);

keV = data_spekcalc{1};
intensity = data_spekcalc{2};