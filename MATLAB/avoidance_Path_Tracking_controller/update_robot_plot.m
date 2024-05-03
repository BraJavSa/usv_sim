function [M1, M2]=update_robot_plot(M1,M2,xrk,yrk,psir,Xd, Yd,obx,oby)
        delete(M1);
        delete(M2);
        M1 = MobilePlot(xrk, yrk, psir);
        xlim([-10, 10]);
        ylim([0, 20]); 
        hold on
        t = linspace(0, 2*pi, 100);
        x_circle1 = obx + 0.5 * cos(t);
        y_circle1 = oby + 0.5 * sin(t);
        x_circle2 = obx + 3 * cos(t);
        y_circle2 = oby + 3 * sin(t);
        plot(x_circle1, y_circle1, 'r'); % Dibujar el primer círculo en rojo
        plot(x_circle2, y_circle2, 'g'); % Dibujar el segundo círculo en verde

        
        M2=plot(Xd, Yd, 'bo', 'MarkerSize', 4); % Plot the point (Xd, Yd) in red
        

end
