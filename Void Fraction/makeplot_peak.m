function makeplot_peak(hObject,event, linescan, direction)
    n = int16(get(hObject,'Value'));

    if direction == "vert"
        findpeaks(linescan{n}.vert, 'Annotate','extents');
        title(['Vertical ', num2str(n)])
    elseif direction == "horiz"
        findpeaks(linescan{n}.horiz, 'MinPeakWidth',1,'MinPeakHeight',0.0125,'Annotate','extents');
        title(['Horizontal ', num2str(n)])
    end
    
    drawnow;
end