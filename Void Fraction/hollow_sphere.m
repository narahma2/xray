function [volume, scan] = hollow_sphere(Ro, phi)
% hollow_sphere - Calculates quantities related to a hollow core sphere
%
% Syntax:  [] = hollow_sphere(Ro, phi)
%
% Inputs:
%    Ro                     Radius of sphere
%    phi                    Void fraction (measure of the size of the hollow core)
%
% Outputs:
%    volume                 Structure containing the volume information (total, core, and solid)
%    scan                   Structure containing the line scan profile through the sphere
%
% Example: 
%    vol = hollow_sphere(Ro, phi);
%    [~, scan] = hollow_sphere(Ro, phi);
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: 
%
% Author: Naveed Rahman, B.Sc. in Mechanical Engineering (2016) at UIUC
% Terrence Meyer Research Group, Zucrow Laboratories, Purdue University
% Email Address: rahmann@purdue.edu  
% January 2018; Last revision: 29-Jan-2018
if phi < 0
    volume = [];
    scan = [];
else
    V = (4/3)*pi*Ro^3;

    X = phi*Ro^3;
    Ri = nthroot(X,3);

    V_A = (4/3)*pi*Ri^3;
    V_W = (4/3)*pi*(Ro^3 - Ri^3);

    volume.air = V_A;
    volume.total = V;
    volume.water = V_W;

    h = linspace(0,Ro);
    i = 1;

    while h(i) < (Ro - Ri)
        LS(i) = 2*sqrt(2*h(i)*Ro - h(i)^2);
        i = i + 1;
    end

    for j = i:length(h)
        LS(j) = 2*(sqrt(2*h(j)*Ro - h(j)^2)) - 2*(sqrt(2*(h(j)-h(i))*Ri - (h(j)-h(i))^2));
    end

    scan.profile = [LS, fliplr(LS)];
    scan.x = linspace(-Ro, Ro, 2*length(h));
end