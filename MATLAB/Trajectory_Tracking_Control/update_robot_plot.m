function [M1, M2]=update_robot_plot(M1,M2,xrk,yrk,psir,Xd, Yd)
        delete(M1);
        delete(M2);
        M1 = MobilePlot(xrk, yrk, psir);
        hold on
        M2=plot(Xd, Yd, 'bo', 'MarkerSize', 4); % Plot the point (Xd, Yd) in red
end
