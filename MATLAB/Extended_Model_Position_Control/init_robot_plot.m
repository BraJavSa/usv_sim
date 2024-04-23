function M1=init_robot_plot(xrk,yrk,psir,Xd,Yd)
    % Plot the desired position
    fig1 = figure('Name','Robot Movement');
    set(fig1,'position',[60 60 980 600]);
    axis square; cameratoolbar
    axis([-50 50 -50 50 0 1]);
    grid on
    MobileRobot;
    M1 = MobilePlot(xrk, yrk, psir);
    hold on
    plot(Xd,Yd,'r','LineWidth',2);
    hold on
end
