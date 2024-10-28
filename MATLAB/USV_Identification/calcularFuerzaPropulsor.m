function T = calcularFuerzaPropulsor(cmd)
    % Parámetros para el caso de cmd > 0.01
    A_pos = 0.01;
    K_pos = 59.82;
    B_pos = 5.0;
    v_pos = 0.38;
    C_pos = 0.56;
    M_pos = 0.28;
    maxForceFwd = 100.0;  % Fuerza máxima hacia adelante

    % Parámetros para el caso de cmd < 0.01
    A_neg = -199.13;
    K_neg = -0.09;
    B_neg = 8.84;
    v_neg = 5.34;
    C_neg = 0.99;
    M_neg = -0.57;
    maxForceRev = -100.0;  % Fuerza máxima hacia atrás

    % Calcular la fuerza en función del comando de empuje (cmd)
    if cmd > 0.01
        % Caso positivo
        T = A_pos + (K_pos - A_pos) / ( (C_pos + exp(-B_pos * (cmd - M_pos)))^(1 / v_pos) );
    elseif cmd < -0.01
        % Caso negativo
        T = A_neg + (K_neg - A_neg) / ( (C_neg + exp(-B_neg * (cmd - M_neg)))^(1 / v_neg) );
    else
        % Caso neutro
        T = 0;
    end

   if  T > 100
        % Caso positivo
        T = 100;
    elseif T < -100
        % Caso negativo
        T = -100;
   end
end
