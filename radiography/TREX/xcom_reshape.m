function xcom_I = xcom_reshape(xcom_keV, xcom_I, source_keV)
%XCOM_RESHAPE   Linear interpolation to reconfigure the X-axis of the XCOM
%               spectra to match SpekCalc/XOP input.
%
% Example:
%               xcom_I = xcom_reshape(xcom_keV, xcom_att, source_keV);
%
% Inputs:
%               xcom_keV:       XCOM spectra keV.
%               xcom_I:         XCOM spectra structure (keV and att. coeffs.)
%               source_keV:     X-axis energy in keV of the source spectra.
%
% Output:
%               xcom_I:         Reshaped XCOM attenuation coefficients.
%
% Author: Naveed Rahman
% Creation Date: 12 May 2020
% Credits:

% Reshape the XCOM spectra
xcom_I = structfun(@(x) interp1(xcom_keV, x, source_keV, 'linear', ...
                                'extrap'), ...
                   xcom_I, 'UniformOutput', false);
