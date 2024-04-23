function [Uref, Wref,error, xrk, yrk, psir] = controller(Xd, Yd, Xdp, Ydp,Xr, Yr, euler_angles)
  
        xrk=Xr;
        yrk=Yr;
        psir=euler_angles(1);
        kx=1;
        ky=1;
        a=0.2; 
        xre = Xd - xrk;
        yre = Yd - yrk;
           
        vx= Xdp+kx*(xre);
        vy= Ydp+ky*(yre);
        vpsi=-(vx/a)*sin(psir)+(vy/a)*cos(psir);
        Uref=vx*cos(psir)+vy*sin(psir);
        Wref=vpsi;
        error = sqrt(xre^2 + yre^2);
        if error <=0.5
            Uref=0;
            Wref=0;
        end
end

