function I = beer_lambert(I0, atten_coeff, epl)
%BEER_LAMBERT           Returns output of Beer-Lambert law.
%
% Example:
%           I = beer_lambert(I0, atten_coeff, epl);
%
% Inputs:
%           I0:             Incident intensity.
%           atten_coeff:    Attenuation coefficient of material [Units: 1/L].
%           epl:            Path length of material [Units: L].
%
% Outputs:
%           I:              Transmitted intensity from Beer-Lambert.
%
% Author:           Naveed Rahman
% Creation Date:    5 June 2020

I = I0 .* exp(-atten_coeff * epl);