clc; clear all; close all;

% Cargar los datos de torques y tiempo desde el archivo .mat
load('sim_val.mat');
load('delta_valores4.mat', 'delta');

% Asignar los valores de delta a variables individuales
delta_1 = delta(1);
delta_2 = delta(2);
delta_3 = delta(3);
delta_4 = delta(4);
delta_5 = delta(5);
delta_6 = delta(6);
delta_7 = delta(7);
delta_8 = delta(8);
delta_9 = delta(9);
delta_10 = delta(10);
delta_11 = delta(11);

% Definición de matrices del sistema
M = [delta_1, 0, 0;
     0, delta_2, delta_3;
     0, delta_3, delta_4];

D = [delta_8, 0, 0;
     0, delta_9, delta_10;
     0, delta_10, delta_11];

% Condiciones iniciales de velocidad
u = zeros(length(t), 1);
v = zeros(length(t), 1);
r = zeros(length(t), 1);
% Torques de entrada (asumimos T_v = 0, ya que solo están T_u y T_r)
T_v = zeros(length(t), 1);
IM = inv(M);

% Parámetros para la optimización
initial_factor = 1.01; % Valor inicial del factor de suavidad
options = optimset('Display', 'iter', 'TolFun', 1e-4); % Opciones de optimización

% Optimización para encontrar el mejor factor de suavidad
optimal_factor = fminunc(@(f) optimizacionSaturacion(f, T_u, T_r, T_v, t, IM, D, delta, vel_u, vel_v,vel_r), initial_factor, options);

% Simulación final con el factor óptimo
ts = t(2) - t(1); % Suponiendo que t es un vector de tiempo equidistante
for k = 1:length(t) - 1
    % Estado de velocidades actuales
    vel = [u(k); v(k); r(k)];
    
    % Cálculo del vector C(vel) * vel
    C = [0, -delta_5 * r(k), - delta_6 * v(k) - delta_3 * r(k);
         delta_5 * r(k), 0, delta_7 * u(k);
         delta_6 * v(k) + delta_3 * r(k), -delta_7 * u(k), 0];

    % Cálculo de las derivadas de velocidad usando: M * d_vel = T - C*vel - D*vel
    d_vel = IM * ([T_u(k); T_v(k); T_r(k)] - C * vel - D * vel);
    
    % Integración de Euler para obtener las velocidades en el siguiente instante
    u(k + 1) = u(k) + d_vel(1) * ts;
    u(k + 1) = saturacionTanHiperbolica(u(k + 1), optimal_factor);
    v(k + 1) = v(k) + d_vel(2) * ts;
    r(k + 1) = r(k) + d_vel(3) * ts;
end

% Gráficas de resultados (sin cambios)
figure;
subplot(3, 1, 1); 
plot(t, u, 'r', 'DisplayName', 'u modelada');
hold on;
plot(t, vel_u, 'b', 'DisplayName', 'u medida');
xlabel('Tiempo (s)');
ylabel('Velocidad');
title('VEL U');
legend;
grid on;

subplot(3, 1, 2); 
plot(t, r, 'r', 'DisplayName', 'r modelada');
hold on;
plot(t, vel_r, 'b', 'DisplayName', 'r medida');
xlabel('Tiempo (s)');
ylabel('Velocidad');
title('VEL R');
legend;
grid on;

subplot(3, 1, 3); 
plot(t, v, 'r', 'DisplayName', 'v modelada');
hold on;
plot(t, vel_v, 'b', 'DisplayName', 'v medida');
xlabel('Tiempo (s)');
ylabel('Velocidad');
title('VEL V');
legend;
grid on;

sgtitle('Resultados de Velocidades')

error_u = calcularECM(u, vel_u);
error_v = calcularECM(v, vel_v);
error_r = calcularECM(r, vel_r);

fprintf('Error cuadrático medio de u: %.4f\n', error_u);
fprintf('Error cuadrático medio de v: %.4f\n', error_v);
fprintf('Error cuadrático medio de r: %.4f\n', error_r);

function ecm = calcularECM(vector1, vector2)
    if length(vector1) ~= length(vector2)
        error('Los vectores deben tener el mismo tamaño');
    end
    ecm = mean((vector1 - vector2).^2);
end

function salida = saturacionTanHiperbolica(entrada, factorSuavidad)
    limite = 2; % valor de saturación
    salida = limite * tanh(entrada * factorSuavidad / limite);
end

function error = optimizacionSaturacion(factor, T_u, T_r, T_v, t, IM, D, delta,vel_u, vel_v,vel_r)
    % Inicialización de velocidades
    u = zeros(length(t), 1);
    v = zeros(length(t), 1);
    r = zeros(length(t), 1);
    ts = t(2) - t(1); % Suponiendo que t es un vector de tiempo equidistante

    % Simulación para calcular el ECM con el factor de suavidad actual
    for k = 1:length(t) - 1
        vel = [u(k); v(k); r(k)];

        C = [0, -delta(5) * r(k), - delta(6) * v(k) - delta(3) * r(k);
             delta(5) * r(k), 0, delta(7) * u(k);
             delta(6) * v(k) + delta(3) * r(k), -delta(7) * u(k), 0];

        d_vel = IM * ([T_u(k); 0; T_r(k)] - C * vel - D * vel);

        u(k + 1) = u(k) + d_vel(1) * ts;
        u(k + 1) = saturacionTanHiperbolica(u(k + 1), factor);
        v(k + 1) = v(k) + d_vel(2) * ts;
        r(k + 1) = r(k) + d_vel(3) * ts;
    end

    % Calcula el ECM para u
    error = calcularECM(u, vel_u);
end
