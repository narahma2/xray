function makeplot_manual(hObject,~, h,hh, diameter,linescan, voidfr, slices, name)
    n = int16(get(hObject,'Value'));

    h = plot(linescan{n}.horiz_x, linescan{n}.horiz);
    hold on

    [~, s] = hollow_sphere(diameter{n}.X/2, voidfr(n));
    hh = plot(s.x, s.profile, 'r');

    legend('Horizontal Line Scan',['\phi = ', num2str(voidfr(n))])
    title('Line Scan Idealization - Horizontal')
    xlabel('Feature Width [cm]')
    ylabel('EPL [cm]')
    title(['Line Scan Idealization: Slice ', num2str(slices(n)), ' Feature ', name, ': Index ', num2str(n)])
    hold off
    
    drawnow;
end