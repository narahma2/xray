function [keV, mu, density] = nist_xcom(materials, att_type, units, max_keV)
%NIST_XCOM      Function that will read in the corresponding material
%               attenuation coefficient with massaged out edge values.
%
% Example:
%               [keV, att_coeff] = nist_xcom({'Air', 'Al'}, 8, 'cm', 150);
%
% Inputs:
%               materials:  Structure of materials (see NIST XCOM DATA.xlsx).
%               att_type:   Type of attenuation (2-8): 2=coh. scatter, 
%                           3=incoh. scatter, 4=photoel. abs., 5=nucl. pp.,
%                           6=el. pp., 7=tot. w. coh., 8=tot. w/o coh.
%               units:      EPL units to be used: 'um', 'mm', 'cm'
%               max_keV:    Maximum keV for the output (optional, def.=150)
%
% Outputs:
%               keV:        X-axis values converted to keV (edges are slightly
%                           shifted for calculations).
%               mu:         Attenuation coeff. in desired units (def.=1/cm).
%               density:    Density of material in g/cm^3
%                           (see NIST XCOM DATA.xlsx)
%
% Dependencies:
%               "NIST XCOM DATA.xlsx":      Partial NIST database.
%
% Author:           Naveed Rahman
% Creation Date:    5 June 2020
% Credits:          Ben Halls

% Number of materials requested
ms = max(size(materials));

% Generates Common Energy Axis including all k-edges
MeV = 1;
for i = 1:ms
    tmp = xlsread('NIST XCOM DATA', materials{i});
    MeV = vertcat(MeV, tmp(:, 1));
end

MeV = unique(MeV);
MeVcom = MeV;

% Generates Attenuation Coefficients, interpolated to common energy axis
for i = 1:ms
    tmp = xlsread('NIST XCOM DATA', materials{i});
    
    % Load photoelectric absorption if scintillator
    if strcmp(materials{i}, 'CsI') || strcmp(materials{i}, 'LYSO')
        acs = tmp(:, 4);
    else
        acs = tmp(:, att_type);
    end
    
    % Load density from spreadsheet
    rho = tmp(1, 9);
    
    % Attenuation coefficient in units of 1/cm        
    m = rho .* acs;
    
    % Get energy axis
    MeV = tmp(:, 1);

    % Adds a small offset to duplicate numbers
    for j = 2:max(size(MeV))
        if MeV(j) == MeV(j - 1)
            MeV(j) = MeV(j) + 0.000001;
        end
    end
    
    mq = interp1(MeV, m, MeVcom);
    newstruct = genvarname(materials{i})l
    mu.(newstruct) = mq;
    density.(newstruct) = rho;
end

% Convert energy axis to keV
keV = MeVcom * 1000;

% Re-scale the attenuation coefficient to match the EPL values
if units == 'um':
    mu == structfun(@(x) x/10000, mu, 'UniformOutput', false);
elseif units == 'mm':
    mu == structfun(@(x) x/10, mu, 'UniformOutput', false);
elseif units == 'cm':
    % No change needed
end

% Default max_keV to 150 if no input provided
if nargin < 4
    max_keV = 150;
end

% Cut down the array to the maximum requested keV
[~, idx] = min(abs(keV - max_keV));
mu = structfun(@(x) x(1:idx), mu, 'UniformOutput', false);
keV = keV(1:idx);