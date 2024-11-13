clc;
clear all;
close all;

% Cargar los datos de torques y tiempo desde el archivo .mat
load('ident_usv.mat');
load('delta_valores_4_2.mat', 'delta');

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

% Parámetro lambda
landa = 100000; %lambda
F1 = tf(landa, [1 landa]);

% Simulación de las señales de velocidad
u_f = lsim(F1, vel_u, t)';
v_f = lsim(F1, vel_v, t)';
r_f = lsim(F1, vel_r, t)';

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
T_v = zeros(length(t), 1);  % Torques de entrada (asumimos T_v = 0)
IM = inv(M);  % Inversa de M

% Simulación
for k = 1:length(t) - 1
    % Estado de velocidades actuales
    vel = [u(k); v(k); r(k)];
    
    % Cálculo del vector C(vel) * vel
    C = [0, -delta_5 * r(k), - delta_6 * v(k) - delta_3 * r(k);
         delta_5 * r(k), 0, delta_7 * u(k);
         delta_6 * v(k) + delta_3 * r(k), -delta_7 * u(k), 0];
     
    % Cálculo de las derivadas de velocidad usando: M * d_vel = T - C * vel - D * vel
    d_vel = IM * ([T_u(k); T_v(k); T_r(k)] - C * vel - D * vel);
    
    % Integración de Euler para obtener las velocidades en el siguiente instante
    u(k + 1) = u(k) + d_vel(1) * ts;
    v(k + 1) = v(k) + d_vel(2) * ts;
    r(k + 1) = r(k) + d_vel(3) * ts;
end

% Entrenamiento de la red neuronal para corregir el error
% (Ejemplo: red neuronal de regresión para corregir el error de u)
net_u = feedforwardnet(10);  % Red neuronal con 10 neuronas en la capa oculta
net_u = train(net_u, t', u - vel_u);  % Entrenamiento de la red neuronal

% Corrección del error usando la red neuronal entrenada
corrected_u = u - vel_u + net_u(t');  % Ajuste de la variable u

% Asegurarnos de que t y corrected_u tengan la misma longitud
if length(t) > length(corrected_u)
    corrected_u = [corrected_u; zeros(length(t) - length(corrected_u), 1)];
elseif length(t) < length(corrected_u)
    corrected_u = corrected_u(1:length(t));
end

% Graficar los resultados
figure;

% Subplot para la velocidad u
subplot(3, 1, 1); % 3 filas, 1 columna, 1ª gráfica
plot(t, corrected_u, 'r', 'DisplayName', 'u modelada corregida');
hold on;
plot(t, vel_u, 'b', 'DisplayName', 'u medida');
xlabel('Tiempo (s)');
ylabel('Velocidad');
title('VEL U');
legend;
grid on;
ylim([-2.8, 2.8]);  % Establecer los límites de la escala en -2.5 a 2.5

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
ylim([-1.5, 1.5]);  % Establecer los límites de la escala en -1.5 a 1.5

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
ylim([-1, 1]);  % Establecer los límites de la escala en -1 a 1
