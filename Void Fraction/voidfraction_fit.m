function voidfraction = voidfraction_fit(diameter, linescan_x, linescan)

linescan_area = trapz(linescan_x, linescan);

minimize = zeros(101, 1);
n = 0;
vect = linspace(0,1,101);

for i = vect
    n = n + 1;
    [~, ls] = hollow_sphere(diameter/2, i);
    fit_area = trapz(ls.x, ls.profile);
    minimize(n) = abs(linescan_area - fit_area);
end

[~, index] = min(minimize);

voidfraction = vect(index);