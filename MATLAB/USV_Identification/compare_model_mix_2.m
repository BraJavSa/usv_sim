clc, clear all, close all;

% Cargar los datos de torques y tiempo desde el archivo .mat
load('sim_val.mat');
load('opt.mat', 'chi');
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

% Definir el intervalo de búsqueda para x entre chi(1) y delta_1
x_min = chi(1);
x_max = delta_1;
T_v=T_u*0;
% Definir la función objetivo para minimizar el ECM de u
f_objetivo = @(x) simularSistema(x, delta, t, T_u, T_v, T_r, vel_u, ts);

% Encontrar el valor óptimo de x usando fminbnd
x_opt = fminbnd(f_objetivo, x_min, x_max);

fprintf('El valor óptimo de x es: %.4f\n', x_opt);

% Simulación final con el valor óptimo de x
[~, u_opt] = simularSistema(x_opt, delta, t, T_u, T_v, T_r, vel_u, ts);

% Gráfica de resultados
figure;
plot(t, u_opt, 'r', 'DisplayName', 'u modelada óptima');
hold on;
plot(t, vel_u, 'b', 'DisplayName', 'u medida');
xlabel('Tiempo (s)');
ylabel('Velocidad');
title('Comparación de u modelada y medida');
legend;
grid on;
ylim([-2.8, 2.8]);

% Función de simulación para calcular el ECM de u
function [ecm, u] = simularSistema(x, delta, t, T_u, T_v, T_r, vel_u, ts)
    % Definición de matrices del sistema con x como variable
    M = [x, 0, 0;
         0, delta(2), delta(3);
         0, delta(3), delta(4)];

    D = [delta(8), 0, 0;
         0, delta(9), delta(10);
         0, delta(10), delta(11)];

    % Condiciones iniciales de velocidad
    u = zeros(length(t),1);
    v = zeros(length(t),1);
    r = zeros(length(t),1);

    % Cálculo de la matriz inversa de M
    IM = inv(M);

    % Simulación
    for k = 1:length(t) - 1
        vel = [u(k); v(k); r(k)];

        % Matriz C para el sistema
        C = [0, -delta(5) * r(k), - delta(6) * v(k)-delta(3)*r(k);
             delta(5)*r(k), 0, delta(7) * u(k);
             delta(6) * v(k)+delta(3)*r(k), -delta(7) * u(k), 0];

        % Cálculo de las derivadas de velocidad
        d_vel = IM * ([T_u(k); T_v(k); T_r(k)] - C * vel - D * vel);

        % Integración de Euler para obtener las velocidades en el siguiente instante
        u(k + 1) = u(k) + d_vel(1) * ts;
        u(k + 1) = saturacion_tanh(u(k + 1));  % Aplicar saturación
        v(k + 1) = v(k) + d_vel(2) * ts;
        r(k + 1) = r(k) + d_vel(3) * ts;
    end

    % Calcular el error cuadrático medio de u
    ecm = calcularECM(u, vel_u);
end

% Función de saturación con tanh
function salida = saturacion_tanh(valor)
    limite = 2;
    salida = limite * tanh(valor / limite);
end

% Función para calcular el error cuadrático medio
function ecm = calcularECM(vector1, vector2)
    if length(vector1) ~= length(vector2)
        error('Los vectores deben tener el mismo tamaño');
    end
    ecm = mean((vector1 - vector2).^2);
end
