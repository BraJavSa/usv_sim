function [errorPlot UrefPlot WrefPlot]=init_plot(Uref,Wref,error,tf)

    % Plot the error and control actions
    fig2 = figure('Name', 'Error and Control Actions');
    set(fig2,'position',[60 60 980 600]);
    subplot(3, 1, 1);
    errorPlot = plot(error, 0, 'b', 'LineWidth', 1);
    xlabel('Time (s)');
    ylabel('Error');
    grid on;
    title('Error Evolution');
    axis([0 tf 0 50]); % Limitar los ejes en el subplot

    subplot(3, 1, 2);
    UrefPlot = plot(Uref, 0, 'b', 'LineWidth', 1);
    xlabel('Time (s)');
    ylabel('Uref');
    grid on;
    title('Uref Evolution');
    axis([0 tf -2 2]); % Limitar los ejes en el subplot

    subplot(3, 1, 3);
    WrefPlot = plot(Wref, 0, 'r', 'LineWidth', 1);
    xlabel('Time (s)');
    ylabel('Wref');
    grid on;
    title('Wref Evolution');
    axis([0 tf -1 1]); % Limitar los ejes en el subplot
    % Update plots
    
end
