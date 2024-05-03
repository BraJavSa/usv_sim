function [Uref, Wref,error, xrk, yrk, psir,w_obs] = controller(Xd, Yd, Xdp, Ydp,Xr, Yr, euler_angles,w_obs_a,w_obs_aa,ts,obx,oby)
  
        xrk=Xr;
        yrk=Yr;
        psir=euler_angles(1);
        kx=1;
        ky=1;
        a=0.2; 
        xre = Xd - xrk;
        yre = Yd - yrk;
        angle=calculate_angle(xrk, yrk, psir,obx,oby);
        fuerza=cal_fuerza(xrk,yrk,obx,oby);
        ft=fuerza*cos(angle);
        fr=fuerza*sin(angle);
        distancia = sqrt((obx - xrk)^2 + (oby - yrk)^2);
        w_obs=impedancia(ft,fr,w_obs_a,w_obs_aa,ts);
        if distancia>3
            w_obs=0;
        end
        
        vx= Xdp+kx*(xre);
        vy= Ydp+ky*(yre);
        vpsi=-(vx/a)*sin(psir)+(vy/a)*cos(psir);
        Uref=vx*cos(psir)+vy*sin(psir);
        Wref=vpsi+w_obs;
        error = sqrt(xre^2 + yre^2);
end

function angle = calculate_angle(Xr, Yr, robot_angle,obx,oby)

% Calculate the difference in x and y coordinates between the robot and the object
    
    dx = obx - Xr;
    dy = oby - Yr;
    
    % Calculate the angle between the robot and the object
    object_angle = atan2(dy, dx);
    
    % Convert the angle to a range from 0 to 2*pi
    if object_angle < 0
        object_angle = object_angle + 2*pi;
    end
    
    % Calculate the difference between the robot angle and the angle to the object
    angle_difference = object_angle - robot_angle;
    
    % Convert the angle difference to a range from -pi to pi
    if angle_difference > pi
        angle_difference = angle_difference - 2*pi;
    elseif angle_difference < -pi
        angle_difference = angle_difference + 2*pi;
    end
    
    % Return the angle difference
    angle = angle_difference;
end

function fuerza = cal_fuerza(xr,yr,obx,oby)
    % Definición de los parámetros
    d_min = 0.5;
    d_max = 3;
    f_max = 18;
    % Calcular el valor de 'b'
    b = f_max / (d_max - d_min);
    distancia = sqrt((obx - xr)^2 + (oby - yr)^2);
    % Calcular el valor de la función para el tiempo 't'
    fuerza = f_max - b * (distancia - d_min);
end

function w_obs=impedancia(ft,fr,w_obs_a,w_obs_aa,ts)
    B = 0.6;
    K = 4;
    I = 3;
    w_obs = (2*I*w_obs_a - I*w_obs_aa + B*w_obs_a - ft*sign(fr)) / (I*(ts^2) + B*ts + K);
end
