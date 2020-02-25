clearvars

load('ligament.mat');

pix2cm = 21/10000;

ligament_minusvoid = ((1-void01.mask).*(1-void02.mask).*(1-void03.mask).*(1-void04.mask).*(1-void05.mask).*(1-void06.mask).*(1-void07.mask).*ligament.mask .* image_slice);
ligament_withvoids = ligament.mask .* image_slice;

ligament_average = mean2(nonzeros(ligament_minusvoid));

void01_filledmean = void01.mask .* ligament_average;
void02_filledmean = void02.mask .* ligament_average;
void03_filledmean = void03.mask .* ligament_average;
void04_filledmean = void04.mask .* ligament_average;
void05_filledmean = void05.mask .* ligament_average;
void06_filledmean = void06.mask .* ligament_average;
void07_filledmean = void07.mask .* ligament_average;

void01_volume = ((4/3)*pi*(void01.position(3) * void01.position(4) * void01.position(3))/8) * pix2cm^3;
void02_volume = ((4/3)*pi*(void02.position(3) * void02.position(4) * void02.position(3))/8) * pix2cm^3;
void03_volume = ((4/3)*pi*(void03.position(3) * void03.position(4) * void03.position(3))/8) * pix2cm^3;
void04_volume = ((4/3)*pi*(void04.position(3) * void04.position(4) * void04.position(3))/8) * pix2cm^3;
void05_volume = ((4/3)*pi*(void05.position(3) * void05.position(4) * void05.position(3))/8) * pix2cm^3;
void06_volume = ((4/3)*pi*(void06.position(3) * void06.position(4) * void06.position(3))/8) * pix2cm^3;
void07_volume = ((4/3)*pi*(void07.position(3) * void07.position(4) * void07.position(3))/8) * pix2cm^3;

voids_volume = void01_volume + void02_volume + void03_volume + void04_volume + void05_volume + void06_volume + void07_volume;
voids_volume_check = (sum(sum(ligament_withvoids)) - sum(sum(ligament_minusvoid))) * pix2cm^2;

voids_mass = voids_volume * 0.001225;

voids_filledmean = void01_filledmean + void02_filledmean + void03_filledmean + void04_filledmean + void05_filledmean + void06_filledmean + void07_filledmean;

ligament_filled = ligament_minusvoid + voids_filledmean;
ligament_filled_volume = sum(sum(ligament_filled)) * pix2cm^2;
ligament_filled_mass = ligament_filled_volume * 1.00;

area_ligament_total = sum(sum(spones(ligament_filled)));
area_ligament_minusvoid = sum(sum(spones(ligament_minusvoid)));
area_voids = sum(sum(spones(voids_filledmean)));

voidfraction_mass = voids_mass ./ ligament_filled_mass
voidfraction_volume = voids_volume ./ ligament_filled_volume
voidfraction_area = area_voids ./ area_ligament_total