% Cargar datos de entrada y salida
load('ident_usv.mat'); % Asegúrate de que 'ident_usv.mat' contenga T_u, T_v, T_r, vel_u, vel_v, vel_r

% Parámetros de la simulación
Fs = 50; % Frecuencia de muestreo (Hz)
dt = 1 / Fs; % Intervalo de muestreo (s)

% Datos de entrada (torques) y salida (velocidades)
inputData = [T_u, T_v, T_r];
outputData = [vel_u, vel_v, vel_r];

% Inicializar matriz para almacenar las velocidades predichas
predicted_velocities = zeros(size(outputData));

% Calcular las velocidades predichas usando las ecuaciones del modelo
for t = 1:size(inputData, 1)
    % Definir las variables X en función de las velocidades y entradas
    X = [
        outputData(t, 1), ... % vel_u (1)
        outputData(t, 2), ... % vel_v (2)
        outputData(t, 3), ... % vel_r (3)
        inputData(t, 1), ...  % T_u (4)
        inputData(t, 2), ...  % T_v (5)
        inputData(t, 3), ...  % T_r (6)
        % Añadir los términos restantes necesarios (X_7, X_8, ..., X_n)
    ];
    
    % Asegúrate de que los índices de X sean correctos
    % Ecuaciones del modelo
    d_vel_1_dt = (-0.133) * X(2) + (0.004) * X(5) + (0.363) * X(1); % Reemplaza X(18) por X(1)
    d_vel_2_dt = (-0.400) * X(5); % Asegúrate de que X(16) esté definido correctamente
    d_vel_3_dt = (-0.109) * X(1); % Reemplaza X(4) por X(1) según el contexto

    % Actualizar las velocidades predichas
    if t > 1
        predicted_velocities(t, 1) = predicted_velocities(t-1, 1) + d_vel_1_dt * dt; % vel_1
        predicted_velocities(t, 2) = predicted_velocities(t-1, 2) + d_vel_2_dt * dt; % vel_2
        predicted_velocities(t, 3) = predicted_velocities(t-1, 3) + d_vel_3_dt * dt; % vel_3
    else
        % Para el primer paso de tiempo, usar las velocidades medidas como inicialización
        predicted_velocities(t, :) = outputData(t, :);
    end
end

% Graficar resultados
figure;
subplot(3, 1, 1);
plot(vel_u, 'b', 'DisplayName', 'Velocidad medida u');
hold on;
plot(predicted_velocities(:, 1), 'r--', 'DisplayName', 'Velocidad predicha u');
title('Comparación de Velocidades Medidas y Predichas (vel_1)');
xlabel('Tiempo (s)');
ylabel('Velocidad (m/s)');
legend;

subplot(3, 1, 2);
plot(vel_v, 'b', 'DisplayName', 'Velocidad medida v');
hold on;
plot(predicted_velocities(:, 2), 'r--', 'DisplayName', 'Velocidad predicha v');
title('Comparación de Velocidades Medidas y Predichas (vel_2)');
xlabel('Tiempo (s)');
ylabel('Velocidad (m/s)');
legend;

subplot(3, 1, 3);
plot(vel_r, 'b', 'DisplayName', 'Velocidad medida r');
hold on;
plot(predicted_velocities(:, 3), 'r--', 'DisplayName', 'Velocidad predicha r');
title('Comparación de Velocidades Medidas y Predichas (vel_3)');
xlabel('Tiempo (s)');
ylabel('Velocidad (rad/s)');
legend;

sgtitle('Comparación de Velocidades Medidas vs Predicciones del Modelo');
