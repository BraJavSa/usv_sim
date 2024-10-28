clc, clear all, close all,
% Cargar los datos de torques y tiempo desde el archivo .mat
load('muestreo_externo.mat');
load('delta_valores_2.mat', 'delta');

% Asignar los valores de delta a variables individuales
delta_1 = delta(1);
delta_2 = delta(2);
delta_3 = delta(3);
delta_4 = delta(4);
delta_5 = delta(5);
delta_6 = delta(6);
delta_7 = delta(7);
delta_8 = delta(8);

% Definición de matrices del sistema
M = [delta_1, 0, 0;
     0, delta_2, delta_3;
     0, delta_3, delta_4];

D = [delta_5, 0, 0;
     0, delta_6, delta_7;
     0, delta_7, delta_8];

% Condiciones iniciales de velocidad
u = zeros(length(t),1);
v = zeros(length(t),1);
r = zeros(length(t),1);
% Torques de entrada (asumimos T_v = 0, ya que solo están T_u y T_r)
T_v = zeros(length(t),1);
IM=inv(M);
% Simulación
for k = 1:length(t) - 1
    % Estado de velocidades actuales
    vel = [u(k); v(k); r(k)];
    
    % Cálculo del vector C(vel) * vel
    C = [0, 0, -delta_2 * v(k) - delta_3 * r(k);
         0, 0, delta_1 * u(k);
         delta_2 * v(k) + delta_3 * r(k), -delta_1 * u(k), 0];
     
    % Cálculo de las derivadas de velocidad usando: M * d_vel = T - C*vel - D*vel
    d_vel = IM*([T_u(k); T_v(k); T_r(k)] - C * vel - D * vel);
    
    % Integración de Euler para obtener las velocidades en el siguiente instante
    u(k + 1) = u(k) + d_vel(1) * ts;
    v(k + 1) = v(k) + d_vel(2) * ts;
    r(k + 1) = r(k) + d_vel(3) * ts;
end

% Gráfica de los resultados en un subplot
figure;

% Subplot para la velocidad u
subplot(3, 1, 1); % 3 filas, 1 columna, 1ª gráfica
plot(t, u, 'r', 'DisplayName', 'u modelada');
hold on;
plot(t, vel_u, 'b', 'DisplayName', 'u medida');
xlabel('Tiempo (s)');
ylabel('Velocidad');
title('VEL U');
legend;
grid on;

% Subplot para la velocidad r
subplot(3, 1, 2); % 3 filas, 1 columna, 2ª gráfica
plot(t, r, 'r', 'DisplayName', 'r modelada');
hold on;
plot(t, vel_r, 'b', 'DisplayName', 'r medida');
xlabel('Tiempo (s)');
ylabel('Velocidad');
title('VEL R');
legend;
grid on;

% Subplot para la velocidad v
subplot(3, 1, 3); % 3 filas, 1 columna, 3ª gráfica
plot(t, v, 'r', 'DisplayName', 'v modelada');
hold on;
plot(t, vel_v, 'b', 'DisplayName', 'v medida');
xlabel('Tiempo (s)');
ylabel('Velocidad');
title('VEL V');
legend;
grid on;

% Ajuste de espacio entre subplots
sgtitle('Resultados de Velocidades') % Título general para todos los subplots

