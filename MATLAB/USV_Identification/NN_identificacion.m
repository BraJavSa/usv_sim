% Cargar los datos del modelo de identificación (entrenamiento)
load('ident_usv.mat');  % Asegúrate de que las variables estén definidas correctamente
T_v=T_u*0;
% Desplazar las velocidades una posición
vel_u = [0; vel_u(1:end-1)];
vel_v = [0; vel_v(1:end-1)];
vel_r = [0; vel_r(1:end-1)];

% Parámetro de suavizado
lambda = 0.1;

% Filtrar las entradas T_u, T_v y T_r
T_u_filtered = filter(lambda, [1, -(1 - lambda)], T_u);
T_v_filtered = filter(lambda, [1, -(1 - lambda)], T_v);
T_r_filtered = filter(lambda, [1, -(1 - lambda)], T_r);

% Filtrar las salidas vel_u, vel_v, vel_r
vel_u_filtered = filter(lambda, [1, -(1 - lambda)], vel_u);
vel_v_filtered = filter(lambda, [1, -(1 - lambda)], vel_v);
vel_r_filtered = filter(lambda, [1, -(1 - lambda)], vel_r);

% Calcular las aceleraciones como la derivada de las velocidades
accel_u = [0; diff(vel_u_filtered)];
accel_v = [0; diff(vel_v_filtered)];
accel_r = [0; diff(vel_r_filtered)];

% Preparar los datos para la red neuronal
inputs = [T_u_filtered, T_v_filtered, T_r_filtered];  % Entradas: tau_u, tau_v, tau_r
outputs = [vel_u_filtered, vel_v_filtered, vel_r_filtered, accel_u, accel_v, accel_r];  % Salidas: vel_u, vel_v, vel_r, accel_u, accel_v, accel_r

% Definir la arquitectura de la red neuronal
hiddenLayerSize = 10;  % Número de neuronas en la capa oculta (puedes experimentar con este valor)
net = fitnet(hiddenLayerSize);

% Configurar la red para el entrenamiento
net.divideParam.trainRatio = 70/100;  % 70% de los datos para entrenamiento
net.divideParam.valRatio = 15/100;    % 15% de los datos para validación
net.divideParam.testRatio = 15/100;   % 15% de los datos para prueba

% Entrenar la red
[net, tr] = train(net, inputs', outputs');  % Transponer para ajustar la dimensión

% Obtener las salidas predichas
outputs_pred = net(inputs');

% Calcular el error de predicción
errors = outputs' - outputs_pred;  % Error entre las salidas reales y las predicciones
meanSquaredError = mean(errors.^2, 2);  % Error cuadrático medio
disp(['Error cuadrático medio: ', num2str(mean(meanSquaredError))]);

% Ahora cargamos los datos de validación de 'muestreo_externo.mat'
load('muestreo_externo.mat');  % Asegúrate de que las variables estén definidas correctamente en este archivo
T_u_ext = T_u;  % Asumimos que 'T_u_ext' es la variable del archivo
T_v_ext = T_v;  % Asumimos que 'T_v_ext' es la variable del archivo
T_r_ext = T_r;
% Filtrar las entradas T_u, T_v y T_r del conjunto de validación
T_u_ext = filter(lambda, [1, -(1 - lambda)], T_u_ext);  % Asumimos que 'T_u_ext' es la variable del archivo
T_v_ext = filter(lambda, [1, -(1 - lambda)], T_v_ext);  % Asumimos que 'T_v_ext' es la variable del archivo
T_r_ext = filter(lambda, [1, -(1 - lambda)], T_r_ext);  % Asumimos que 'T_r_ext' es la variable del archivo

% Filtrar las salidas vel_u, vel_v, vel_r del conjunto de validación
vel_u_ext = filter(lambda, [1, -(1 - lambda)], vel_u);  % Asumimos que 'vel_u_ext' es la variable del archivo
vel_v_ext = filter(lambda, [1, -(1 - lambda)], vel_v);  % Asumimos que 'vel_v_ext' es la variable del archivo
vel_r_ext = filter(lambda, [1, -(1 - lambda)], vel_r);  % Asumimos que 'vel_r_ext' es la variable del archivo

% Calcular las aceleraciones del conjunto de validación
accel_u_ext = [0; diff(vel_u_ext)];
accel_v_ext = [0; diff(vel_v_ext)];
accel_r_ext = [0; diff(vel_r_ext)];

% Preparar los datos de entrada y salida para la validación

inputs_ext = [T_u_ext, T_v_ext, T_r_ext];  % Entradas: tau_u_ext, tau_v_ext, tau_r_ext
outputs_ext = [vel_u_ext, vel_v_ext, vel_r_ext, accel_u_ext, accel_v_ext, accel_r_ext];  % Salidas: vel_u_ext, vel_v_ext, vel_r_ext, accel_u_ext, accel_v_ext, accel_r_ext

% Validar la red en los datos externos
outputs_pred_ext = net(inputs_ext');

% Calcular el error de predicción en los datos de validación
errors_ext = outputs_ext' - outputs_pred_ext;  % Error entre las salidas reales y las predicciones
meanSquaredError_ext = mean(errors_ext.^2, 2);  % Error cuadrático medio
disp(['Error cuadrático medio en validación externa: ', num2str(mean(meanSquaredError_ext))]);

% Graficar la comparación de las salidas reales vs. predichas para el conjunto de validación
figure;
subplot(2,1,1);
plot(vel_u, 'r-', 'DisplayName', 'Velocidades reales (validación)');
hold on;
plot(outputs_pred_ext(1,:), 'b--', 'DisplayName', 'Velocidades predichas (validación)');
legend;
title('Comparación de Velocidades (u) - Validación Externa');

