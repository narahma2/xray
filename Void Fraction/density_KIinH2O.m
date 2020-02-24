function rho = density_KIinH2O(concentration)
% density_KIinH2O - Returns the density of known KI concentration
%
% Syntax: rho = density_KIinH2O(concentration)
%
% Inputs
% ------
%	'concentration'		percent mass concentration of KI in water [0-50]
%
% Outputs
% -------
%	'rho'               density of the KI in water sample		
%
% Examples
% --------
%	den_KI = density_KIinH2O(10);
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none

% Author: Benjamin R. Halls, Ph.D., Air Force Research Laboratory
% Adapted by: Naveed Rahman, B.Sc. in Mechanical Engineering (2016) at UIUC
% Terrence Meyer Research Group, Zucrow Laboratories, Purdue University
% Email Address: rahmann@purdue.edu  
% July 2017; Last revision: 26-July-2017

    conc = [ 0 , 4 , 20 , 26 , 30 , 50 ];
    dens = [ 1 , 1.028 , 1.168 , 1.227 , 1.26 , 1.54 ];
    %plot(conc,dens,'o');
    
    p1 =   9.078e-05; % (5.93e-05, 0.0001223)
    p2 =    0.006199; % (0.004601, 0.007796)
    p3 =       1.002; % (0.9846, 1.019)
    T = 50;
    f = p1*T^2 + p2*T + p3;
    
    rho_00_KI = 1; % Water
    rho_10_KI = p1*10^2 + p2*10 + p3;
    rho_20_KI = p1*20^2 + p2*20 + p3;
    rho_50_KI = p1*50^2 + p2*50 + p3;
    
    rho = p1*(concentration)^2 + p2*(concentration) + p3;