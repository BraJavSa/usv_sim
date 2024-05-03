% Función principal
function main()
    % Función objetivo a minimizar (error en la evasión del obstáculo)
    objective = @(x) simulate_robot(x(1), x(2), x(3));

    % Límites para las variables de ajuste (B, K, I)
    lb = [0, 0, 0]; % Límites inferiores
    ub = [100, 1, 1]; % Límites superiores

    % Configuración de PSO
    options = optimoptions(@particleswarm, 'SwarmSize', 50, 'MaxIterations', 100);

    % Ejecutar PSO
    [x_opt, fval] = particleswarm(objective, 3, lb, ub, options);

    disp('Parámetros optimizados:');
    disp(['B: ', num2str(x_opt(1))]);
    disp(['K: ', num2str(x_opt(2))]);
    disp(['I: ', num2str(x_opt(3))]);
    disp(['Error: ', num2str(fval)]);

    % Función para simular el robot y calcular el error
    function error = simulate_robot(B, K, I)
        % Parámetros de simulación
        ts = 0.1;
        t = 0:ts:45;
        xi = -3;
        yi = 2;
        psir = pi;
        xrd = 20*sin(0.4*t);
        yrd = 20*sin(0.2*t);
        xrdp = 20*cos(0.4*t)*0.4;
        yrdp = 20*cos(0.2*t)*0.2;
        a = 0.2;
        xr = xi + a*cos(psir);
        yr = yi + a*sin(psir);
        kx = 3;
        ky = 2;
        ft = zeros(size(t));
        fr = zeros(size(t));
        w_obs = zeros(size(t));
        oby = 19.9;
        obx = 0;

        % Simulación del control del robot con los parámetros dados
        for k = 1:length(t)
            % Errores de control
            xre = xrd(k) - xr;
            yre = yrd(k) - yr;

            % Ley de control
            angle = calculate_angle(xr, yr, psir, obx, oby);
            fuerza = cal_fuerza(xr, yr, obx, oby, B);
            ft(k) = fuerza*cos(angle);
            fr(k) = fuerza*sin(angle);
            if k > 2
                w_obs(k) = impedancia(ft(k), fr(k), w_obs(k-1), w_obs(k-2), ts, I, B, K);
            end
            distancia = sqrt((obx - xr)^2 + (oby - yr)^2);
            if distancia > 6
                w_obs(k) = 0;
            end
            vx = xrdp(k) + kx*(xre);
            vy = yrdp(k) + ky*(yre);
            vpsi = -(vx/a)*sin(psir) + (vy/a)*cos(psir);
            u = vx*cos(psir) + vy*sin(psir);
            w = vpsi + w_obs(k);

            % Aplicar acciones de control al robot
            xrp = u*cos(psir) - a*w*sin(psir);
            yrp = u*sin(psir) + a*w*cos(psir);

            % Hallar posiciones
            xr = xr + ts*xrp;
            yr = yr + ts*yrp;
            psir = psir + ts*w;
        end

        % Calcular el error (puedes usar alguna métrica relevante)
        % Por ejemplo, puedes calcular el promedio del error en la evasión del obstáculo
        error = mean(abs(xr - xrd) + abs(yr - yrd));
    end
end

% Función para calcular el ángulo entre el robot y el obstáculo
function angle = calculate_angle(Xr, Yr, robot_angle, obx, oby)
    dx = obx - Xr;
    dy = oby - Yr;
    object_angle = atan2(dy, dx);
    if object_angle < 0
        object_angle = object_angle + 2*pi;
    end
    angle_difference = object_angle - robot_angle;
    if angle_difference > pi
        angle_difference = angle_difference - 2*pi;
    elseif angle_difference < -pi
        angle_difference = angle_difference + 2*pi;
    end
    angle = angle_difference;
end

% Función para calcular la fuerza repulsiva entre el robot y el obstáculo
function fuerza = cal_fuerza(xr, yr, obx, oby, B)
    d_min = 0.5;
    d_max = 6;
    f_max = 10;
    b = f_max / (d_max - d_min);
    distancia = sqrt((obx - xr)^2 + (oby - yr)^2);
    fuerza = f_max - b * (distancia - d_min);
end

% Función para calcular la velocidad angular utilizando un modelo de impedancia
function w_obs = impedancia(ft, fr, w_obs_a, w_obs_aa, ts, I, B, K)
    w_obs = (2*I*w_obs_a - I*w_obs_aa + B*w_obs_a - ft*sign(fr)) / (I*(ts^2) + B*ts + K);
end
