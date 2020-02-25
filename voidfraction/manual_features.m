clearvars

% Load feature
name = 'C';
drop = load(['drop', name, '.mat']);
void = load(['void', name, '.mat']);
fields = fieldnames(drop);
fieldsvoid = fieldnames(void);

% Load EPL image
load('image_epl.mat');

voidfr = [];
rim_x = [];
rim_y = [];
drop_volume = [];
void_volume = [];

for i = 1:numel(fields)
    str = strsplit(drop.(fields{i}).slicenumber, '/');
    slices(i) = str2num(str{1});
    dropcenter_x(i) = drop.(fields{i}).position(2);
    dropcenter_y(i) = drop.(fields{i}).position(1);
    dropradius_x(i) = drop.(fields{i}).position(4) / 2;
    dropradius_y(i) = drop.(fields{i}).position(3) / 2;
    [image_droplet{i}, droplinescan{i}, dropdiameter{i}, dropvoid{i}] = droplet_features_ellipsoid(dropradius_x(i),dropradius_y(i), dropcenter_x(i),dropcenter_y(i), (image_epl(:,:,slices(i)) .* drop.(fields{i}).mask));
    
    voidcenter_x(i) = void.(fieldsvoid{i}).position(2);
    voidcenter_y(i) = void.(fieldsvoid{i}).position(1);
    voidradius_x(i) = void.(fieldsvoid{i}).position(4) / 2;
    voidradius_y(i) = void.(fieldsvoid{i}).position(3) / 2;
    [image_void{i}, voidlinescan{i}, voiddiameter{i}, voidvoid{i}] = droplet_features_ellipsoid(voidradius_x(i),voidradius_y(i), voidcenter_x(i),voidcenter_y(i), (image_epl(:,:,slices(i)) .* void.(fieldsvoid{i}).mask));
     
    drop_volume = [drop_volume, dropvoid{i}.radiusvolume];
    void_volume = [void_volume, voidvoid{i}.radiusvolume];
    voidfr = [voidfr, dropvoid{i}.fraction_hough];
    dropdia_x(i) = dropdiameter{i}.X;
    dropdia_y(i) = dropdiameter{i}.Y;
    dropdia_z(i) = dropdiameter{i}.Z;
%     rim_x = [rim_x, droplinescan{i}.rim_horiz];
%     rim_y = [rim_y, droplinescan{i}.rim_vert];
%     void_dia_x(i) = dropdiameter{i}.X - 2*rim_x(i);
%     void_dia_y(i) = dropdiameter{i}.Y - 2*rim_y(i);    
    
end

% void_dia_z = void_dia_x;
% void_volumefromrim = (4/3)*pi*(void_dia_x.*void_dia_y.*void_dia_z)/8;
% voidfr_voidvolrim = 1 - ((drop_volume - void_volumefromrim) ./ drop_volume);
voidfr_voidvol = void_volume ./ drop_volume;

% figure();
% scatter(void_volume, void_volumefromrim);

%%
figure();
i = 1;
plot(droplinescan{i}.horiz_x, droplinescan{i}.horiz)
hold on
plot(droplinescan{i}.vert_x, droplinescan{i}.vert)

%% Plot valleys
plot_number = 1;
direction = 'horiz';
figure();
if direction == 'horiz'
    findpeaks(-droplinescan{plot_number}.horiz, 'MinPeakWidth',2, 'Annotate','extents');
    title(['Horizontal ', num2str(plot_number)])
elseif direction == 'vert'
    findpeaks(-droplinescan{plot_number}.vert, 'MinPeakWidth',2, 'Annotate','extents');
    title(['Vertical ', num2str(plot_number)])
end

g = uicontrol('style','slider','position',[20 20 300 20],'min',1,'max',numel(fields),'Value',1,'Units','Normalized','SliderStep',[1/(numel(fields)-1) 1]);
addlistener(g,'ContinuousValueChange',@(hObject, event) makeplot_valley(hObject,event, droplinescan, direction));

%% Plot peak
plot_number = 1;
direction = 'vert';
figure();
if direction == 'horiz'
    findpeaks(droplinescan{plot_number}.horiz, 'MinPeakWidth',1,'MinPeakHeight',0.0125, 'Annotate','extents');
    title(['Horizontal ', num2str(plot_number)])
elseif direction == 'vert'
    findpeaks(droplinescan{plot_number}.vert, 'Annotate','extents');
    title(['Vertical ', num2str(plot_number)])
end

g = uicontrol('style','slider','position',[20 20 300 20],'min',1,'max',numel(fields),'Value',1,'Units','Normalized','SliderStep',[1/(numel(fields)-1) 1]);
addlistener(g,'ContinuousValueChange',@(hObject, event) makeplot_peak(hObject,event, droplinescan, direction));

%% Plot idealized hollow sphere (horizontal) through fitting void fraction

% for plot_number = 1:length(dropdiameter)
for plot_number = 4
    voidfraction = voidfraction_fit(dropdiameter{plot_number}.X, droplinescan{plot_number}.horiz_x,droplinescan{plot_number}.horiz);
    [v, s] = hollow_sphere(dropdiameter{plot_number}.X/2, voidfraction);

    figure();
    plot(10*droplinescan{plot_number}.horiz_x, 10*droplinescan{plot_number}.horiz);
    hold on
    plot(10*s.x, 10*s.profile, 'r');
    hold off
    legend('Horizontal Line Scan',['\phi = ', num2str(voidfraction)],'Location','best')
    title(['Line Scan Idealization: Slice ', num2str(slices(plot_number)), ' Feature ', name, ': Index ', num2str(plot_number)])
    xlabel('Feature Width [mm]')
    ylabel('EPL [mm]')
%     saveas(gcf, [name, '/IdealizedSphereFit_Slice', num2str(slices(plot_number)),'Feature',name,'Index',num2str(plot_number), '.png'])
%     close(gcf);
end

%% Plot idealized hollow sphere (horizontal)
plot_number = 17;

[v, s] = hollow_sphere(dropdiameter{plot_number}.X/2, voidfr(plot_number));

figure();
h = plot(droplinescan{plot_number}.horiz_x, droplinescan{plot_number}.horiz);
hold on
hh = plot(s.x, s.profile, 'r');
hold off
legend('Horizontal Line Scan',['\phi = ', num2str(voidfr(plot_number))])
title(['Line Scan Idealization: Slice ', num2str(slices(plot_number)), ' Feature ', name, ': Index ', num2str(plot_number)])
xlabel('Feature Width [cm]')
ylabel('EPL [cm]')

g = uicontrol('style','slider','position',[20 20 300 20],'min',1,'max',numel(fields),'Value',1,'Units','Normalized','SliderStep',[1/(numel(fields)-1) 1]);
addlistener(g,'ContinuousValueChange',@(hObject, event) makeplot_manual(hObject,event, h,hh, dropdiameter,droplinescan, voidfr, slices, name));

%% Plot idealized hollow sphere (vertical)
plot_number = 1;

[v, s] = hollow_sphere(hough(plot_number), voidfr(plot_number));

figure();
h = plot(slice_features{index(plot_number), 9}.vert_x, slice_features{index(plot_number), 9}.vert);
hold on
hh = plot(s.x, s.profile);
hold off
legend('Vertical Line Scan',['\phi = ', num2str(voidfr(plot_number))])
title(['Line Scan Idealization: Slice ', num2str(slice_features{index(plot_number),1}), ' Feature ', num2str(slice_features{index(plot_number),2}), ': Index ', num2str(plot_number)])
xlabel('Feature Width [cm]')
ylabel('EPL [cm]')

g = uicontrol('style','slider','position',[20 20 300 20],'min',1,'max',305,'Value',1,'Units','Normalized','SliderStep',[1/(305-1) 1]);
addlistener(g,'ContinuousValueChange',@(hObject, event) makeplot4(hObject,event, h,hh, slice_features,index,hough,voidfr));