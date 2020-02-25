function makeplot_valley(hObject,event, linescan, direction)
    n = int16(get(hObject,'Value'));

    findpeaks(-linescan{n}.horiz, 'MinPeakWidth',2, 'Annotate','extents');
    title(num2str(n))
    
    if direction == "vert"
        findpeaks(-linescan{n}.vert, 'MinPeakWidth',1,'Annotate','extents');
        title(['Vertical ', num2str(n)])
    elseif direction == "horiz"
        findpeaks(-linescan{n}.horiz, 'MinPeakWidth',1,'Annotate','extents');
        title(['Horizontal ', num2str(n)])
    end
    
    drawnow;
end