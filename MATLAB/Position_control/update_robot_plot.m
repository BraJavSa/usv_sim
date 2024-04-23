function M1=update_robot_plot(M1,xrk,yrk,psir)
        delete(M1);
        M1 = MobilePlot(xrk, yrk, psir);
        hold on
end
