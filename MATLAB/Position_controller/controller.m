function [Uref, Wref,error, xrk, yrk, psir] = controller(Xd, Yd, Xr, Yr, euler_angles)
    
    % Controller gains
    ku = 1.8; 
    kw = 1.5; 
    ks = 0.08; % Saturation gain
    xrk=Xr;
    yrk=Yr;
    psir=euler_angles(1);
    % Calculate position error
    Xe = Xd - Xr;
    Ye = Yd - Yr;
    error = sqrt(Xe^2 + Ye^2);
    a_e = atan2(Ye, Xe);
    angular_difference = normalizeAngle(normalizeAngle(a_e) - normalizeAngle(euler_angles(1)));

    % Define control actions
    if error <= 0.2 
        Uref = 0;
        Wref = 0;          
    else
        Uref = ku * tanh(ks * error) * cos(angular_difference);
        Wref = kw * angular_difference + ku * (tanh(ks * error) / error) * sin(angular_difference) * cos(angular_difference);
    end

    if abs(Wref) > 0.7
        Wref = sign(Wref) * 0.7;
    end
end

