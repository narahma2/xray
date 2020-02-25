clearvars

drop = load('dropB.mat');
void = load('voidB.mat');
load('image_epl.mat');

pix2cm = 21/10000;

fieldsdrop = fieldnames(drop);
fieldsvoid = fieldnames(void);

for i = 1:numel(fieldsdrop)
    str = strsplit(drop.(fieldsdrop{i}).slicenumber, '/');
    slices(i) = str2num(str{1});
    drop_minusvoid{i} = (1 - void.(fieldsvoid{i}).mask) .* drop.(fieldsdrop{i}).mask .* image_epl(:,:,slices(i));
    drop_withvoids{i} = drop.(fieldsdrop{i}).mask .* image_epl(:,:,slices(i));
    drop_average(i) = mean2(nonzeros(drop_minusvoid{i}));

    void_filledmean{i} = void.(fieldsvoid{i}).mask .* drop_average(i);
    void_volume(i) = ((4/3)*pi*(void.(fieldsvoid{i}).position(3) * void.(fieldsvoid{i}).position(4) * void.(fieldsvoid{i}).position(3))/8) * pix2cm^3;
    void_volume_check(i) = (sum(sum(drop_withvoids{i})) - sum(sum(drop_minusvoid{i}))) * pix2cm^2;
    void_mass(i) = void_volume(i) * 0.001225;

    drop_volume(i) = sum(sum(drop_withvoids{i})) * pix2cm^2;
    drop_filled{i} = drop_minusvoid{i} + void_filledmean{i};
    drop_filled_volume(i) = sum(sum(drop_filled{i})) * pix2cm^2;
    drop_volume_check(i) = ((4/3)*pi*(drop.(fieldsdrop{i}).position(3) * drop.(fieldsdrop{i}).position(4) * drop.(fieldsdrop{i}).position(3))/8) * pix2cm^3;
    drop_filled_mass(i) = drop_filled_volume(i) * 1.00;
    drop_mass_check(i) = drop_volume_check(i) * 1.00;
    
    area_drop_total(i) = sum(sum(spones(drop_withvoids{i})));
    area_drop_minusvoid(i) = sum(sum(spones(drop_minusvoid{i})));
    area_void(i) = sum(sum(spones(void_filledmean{i})));
    
    voidfraction_mass(i) = void_mass(i) ./ drop_filled_mass(i);
    voidfraction_mass_check(i) = void_mass(i) ./ drop_mass_check(i);
    voidfraction_volume(i) = 1 - (drop_volume(i) ./ drop_volume_check(i));
    voidfraction_volume_filled(i) = void_volume(i) ./ drop_filled_volume(i);
    voidfraction_volume_check(i) = void_volume_check(i) ./ drop_volume_check(i);
    voidfraction_area(i) = area_void(i) ./ area_drop_total(i);
end

[v, s1] = hollow_sphere(pix2cm*drop.(fieldsdrop{1}).position(3)/2, voidfraction_volume_check(1));
[v, s6] = hollow_sphere(pix2cm*drop.(fieldsdrop{6}).position(3)/2, voidfraction_volume_check(6));

% Manually create the line profiles with imtool3D(drop_withvoids{i})
figure(); plot((ss1.cx - mean(ss1.cx))*pix2cm, ss1.profile)
hold on
plot(s1.x, s1.profile)
legend('EPL','Ideal Sphere')
title('Line Scan through Feature 2')
xlabel('Width [cm]')
ylabel('EPL [cm]')

figure(); plot((ss6.cy - mean(ss6.cy))*pix2cm, ss6.profile)
hold on
plot(s6.x, s6.profile)
legend('EPL','Ideal Sphere')
title('Line Scan through Feature 2')
xlabel('Width [cm]')
ylabel('EPL [cm]')

% radiusx = drop.(fieldsdrop{1}).position(4) / 2;
% radiusy = drop.(fieldsdrop{1}).position(3) / 2;
% centerx = drop.(fieldsdrop{1}).position(2);
% centery = drop.(fieldsdrop{1}).position(1);
% 
% [~, linescan] = droplet_features_ellipsoid(radiusx,radiusy, centerx, centery, drop_withvoids{1})
