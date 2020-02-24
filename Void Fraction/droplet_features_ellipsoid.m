function [image_droplet, linescan, diameter, void] = droplet_features_ellipsoid(radiusX,radiusY, centerX, centerY, image_droplet)
% droplet_features - Calculates quantitative features of droplets from an X-ray radiograph
%
% Syntax:  [mass] = droplet_features(radiusX,radiusY, centerX,centerY, image)
%
% Inputs:
%    radiusX                Radius of detected droplet (in pixels) in X
%    radiusY                Radius of detected droplet (in pixels) in Y
%    centerX                X-location of droplet in the image (if using viscircles, X and Y are swapped--use column 2!)
%    centerY                Y-location of droplet in the image (if using viscircles, X and Y are swapped--use column 1!)
%    image                  Double array that contains the image matrix (see readTIFF)
%
% Outputs:
%    mass                   Mass of droplet (in grams)
%    output2 - Description
%
% Example: 
%    Line 1 of example
%    Line 2 of example
%    Line 3 of example
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: readTIFF
%
% Author: Naveed Rahman, B.Sc. in Mechanical Engineering (2016) at UIUC
% Terrence Meyer Research Group, Zucrow Laboratories, Purdue University
% Email Address: rahmann@purdue.edu  
% December 2017; Last revision: 4-Dec-2017

centerXr = round(centerX);
centerYr = round(centerY);
radiusXr = round(radiusX);
radiusYr = round(radiusY);

% Converting 1 pixel to cm equivalent
pix2microns = 21;
pix2cm = (pix2microns)/10000;

%% Line Scans
% Horizontal Line Scan
posX = centerXr + radiusXr + 5;
negX = centerXr - radiusXr - 5;

if negX <= 0
    negX = 1;
end

if posX >= 58
    posX = 58;
end

linescan.horiz = image_droplet(negX:posX, centerYr);
linescan.horiz_x = negX:posX;
linescan.horiz_x = pix2cm*(linescan.horiz_x - median(linescan.horiz_x));

% Vertical Line Scan
posY = centerYr + radiusYr + 5;
negY = centerYr - radiusYr - 5;

if negY <= 0
    negY = 1;
end

if posY >= 384
    posY = 384;
end

linescan.vert = image_droplet(centerXr, negY:posY);
linescan.vert_x = negY:posY;
linescan.vert_x = pix2cm*(linescan.vert_x - median(linescan.vert_x));

% figure();
% plot(linescan.horiz)
% hold on
% plot(linescan.vert)
% hold off
% title('Line Scan Comparison')
% legend('Horizontal Scan', 'Vertical Scan')

% Diagonal Line Scan
for i = -radiusXr:radiusXr
    indX = centerXr+i;
    if indX <= 0
        indX = 1;
    end
    if indX >= 58
        indX = 58;
    end
    indY1 = centerYr-i;
    indY2 = centerYr+i;
    if indY1 <= 0
        indY1 = 1;
    end
    if indY1 >= 384
        indY1 = 384;
    end
    if indY2 <= 0
        indY2 = 1;
    end
    if indY2 >= 384
        indY2 = 384;
    end
    linescan.diagneg(i+radiusXr+1) = image_droplet(indX, indY1);         % Positive slope of one
    linescan.diagpos(i+radiusXr+1) = image_droplet(indX, indY2);         % Negative slope of one
end

% Rim calculation from horizontal line scan
% [~,~,w_horiz] = findpeaks(linescan.horiz,'Annotate','extents','MinPeakHeight',0.70*max(linescan.horiz),'WidthReference','halfheight');
% if length(w_horiz) == 1
%     linescan.rim_horiz = pix2cm*mean(w_horiz);
% elseif isempty(w_horiz)
%     linescan.rim_horiz = NaN;
% else
%     linescan.rim_horiz = pix2cm*mean([w_horiz(1),w_horiz(end)]);
% end
% 
% % Rim calculation from vertical line scan
% [~,~,w_vert] = findpeaks(linescan.vert,'Annotate','extents','MinPeakHeight',0.70*max(linescan.vert),'WidthReference','halfheight');
% if length(w_vert) == 1
%     linescan.rim_vert = pix2cm*mean(w_vert);
% elseif isempty(w_vert)
%     linescan.rim_vert = NaN;
% else
%     linescan.rim_vert = pix2cm*mean([w_vert(1),w_vert(end)]);
% end

%% Diameters
diameter.center = image_droplet(centerXr, centerYr);                                                    % Diameter from center EPL (in cm)

diameter.max = max(max(image_droplet));                                                                 % Diameter from maximum EPL (in cm)

%xs = sort(image_droplet(:),'descend');
%diameter.line = image((centerXr-round(radiusr/2)):(centerXr+round(radiusr/2)), centerYr);              % Diameters from EPL around the center (in cm)
diameter.mean = mean(linescan.horiz);                                                                   % Diameter from median of EPL line (in cm)

diameter.X = pix2cm*radiusX*2;                                                                          % Diameter from ellipsoid in X
diameter.Y = pix2cm*radiusY*2;                                                                          % Diameter from ellipsoid in Y
diameter.Z = pix2cm*radiusX*2;                                                                          % Diameter from ellipsoid (assuming X = Z)

try
    diameter.ls_horiz = pix2cm*fwhm(negX:posX, linescan.horiz);                                         % Diameter from horizontal line scan width
catch
    diameter.ls_horiz = NaN;
end

try
    diameter.ls_vert = pix2cm*fwhm(negY:posY, linescan.vert);                                           % Diameter from vertical line scan width
catch
    diameter.ls_vert = NaN;
end

% disp(['Center Diameter: ', num2str(diameter.center)])
% disp(['Max Diameter: ', num2str(diameter.max)])
% disp(['Average Diameter: ', num2str(diameter.avg)])
% disp(['Hough Diameter: ', num2str(diameter.hough)])

%% Void Fraction
void.eplvolume = sum(sum(image_droplet)) * pix2cm^2;                                        % volume in cm^3 based on EPL summation
diameter.eplvolume = void.eplvolume / ((4/3)*pi*diameter.X*diameter.Y);                     % characteristic diameter based on the EPL volume
void.radiusvolume = (4/3)*pi*(diameter.X*diameter.Y*diameter.Z)/8;                          % volume in cm^3 based on ellipsoid radius
void.eplmass = void.eplvolume * density_KIinH2O(0);                                         % mass of droplet in grams based on EPL
void.radiusmass = void.radiusvolume * density_KIinH2O(0);                                   % mass of droplet in grams based on Hough radius
void.fraction_hough = 1 - (void.eplmass / void.radiusmass);                                 % void fraction of the droplet based on Hough radius