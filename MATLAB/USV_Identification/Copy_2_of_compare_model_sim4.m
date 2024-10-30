% Cargar los datos de identificación
load('muestreo_externo.mat');

% Parámetros del tiempo de muestreo
dt = 0.02;
time = (0:length(vel_u)-1) * dt;

% Definir las ecuaciones del modelo actualizado con las fuerzas de entrada
x0_model = zeros(size(vel_u));
x1_model = zeros(size(vel_v));
x2_model = zeros(size(vel_r));

% Valores iniciales para las velocidades modeladas
x0_model(1) = vel_u(1);
x1_model(1) = vel_v(1);
x2_model(1) = vel_r(1);

% Simulación del modelo con fuerzas de entrada
for k = 1:length(time)-1
    % Ecuaciones del modelo sindy basado en las obtenidas, incluyendo las fuerzas
    x0_dot = 7355820825260.545 - 15.018 * sin(x0_model(k)) + 5.722 * cos(x0_model(k)) ...
             - 3760487361278.699 * cos(x1_model(k)) - 3.231 * sin(x2_model(k)) ...
             - 5.518 * cos(x2_model(k)) + 18.668 * sin(2 * x0_model(k)) ...
             + 5.344 * cos(2 * x0_model(k)) - 3595333463981.996 * cos(2 * x1_model(k)) ...
             + 2.027 * sin(2 * x2_model(k)) - 4.353 * cos(2 * x2_model(k)) ...
             + T_u(k);  % Añadir la fuerza T_u como entrada

    x1_dot = -247147691554.481 + 0.015 * cos(x0_model(k)) - 129273495881.094 * cos(x1_model(k)) ...
             + 0.572 * sin(2 * x0_model(k)) - 0.081 * cos(2 * x0_model(k)) ...
             + 376421187435.791 * cos(2 * x1_model(k)) - 0.396 * cos(2 * x2_model(k)) ...
             + T_v(k);  % Añadir la fuerza T_v como entrada

    x2_dot = 9440626235905.525 + 0.183 * sin(x0_model(k)) + 0.555 * cos(x0_model(k)) ...
             - 4826294235004.858 * cos(x1_model(k)) + 1.071 * sin(x2_model(k)) ...
             + 1.499 * cos(x2_model(k)) + 0.402 * sin(2 * x0_model(k)) ...
             + 0.163 * cos(2 * x0_model(k)) - 4614332000902.160 * cos(2 * x1_model(k)) ...
             + 1.569 * sin(2 * x2_model(k)) + 0.562 * cos(2 * x2_model(k)) ...
             + T_r(k);  % Añadir la fuerza T_r como entrada

    % Actualización de las velocidades modeladas usando Euler
    x0_model(k+1) = x0_model(k) + x0_dot * dt;
    x1_model(k+1) = x1_model(k) + x1_dot * dt;
    x2_model(k+1) = x2_model(k) + x2_dot * dt;
end

% Comparación gráfica
figure;

subplot(3,1,1);
plot(time, vel_u, 'b', 'DisplayName', 'Real Vel. u');
hold on;
plot(time, x0_model, 'r--', 'DisplayName', 'Model Vel. u');
xlabel('Time (s)');
ylabel('Vel. u');
title('Comparison of Real and Modeled Velocities (u)');
legend;

subplot(3,1,2);
plot(time, vel_v, 'b', 'DisplayName', 'Real Vel. v');
hold on;
plot(time, x1_model, 'r--', 'DisplayName', 'Model Vel. v');
xlabel('Time (s)');
ylabel('Vel. v');
title('Comparison of Real and Modeled Velocities (v)');
legend;

subplot(3,1,3);
plot(time, vel_r, 'b', 'DisplayName', 'Real Vel. r');
hold on;
plot(time, x2_model, 'r--', 'DisplayName', 'Model Vel. r');
xlabel('Time (s)');
ylabel('Vel. r');
title('Comparison of Real and Modeled Velocities (r)');
legend;

sgtitle('Comparison of Real and Modeled Velocities');
