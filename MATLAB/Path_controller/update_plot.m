function [errorPlot UrefPlot WrefPlot]=update_plot(errorPlot,UrefPlot,WrefPlot,error,Uref,Wref,t,k)
         set(errorPlot, 'XData', t(1:k), 'YData', error(1:k));
        set(UrefPlot, 'XData', t(1:k), 'YData', Uref(1:k));
         set(WrefPlot, 'XData', t(1:k), 'YData', Wref(1:k));
end
