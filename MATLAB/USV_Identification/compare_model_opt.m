clc, clear all, close all,
% Cargar los datos de torques y tiempo desde el archivo .mat
load('muestreo_externo.mat');
load('opt.mat', 'chi');

% Asignar los valores de delta a variables individuales
delta_1 = chi(1);
delta_2 = chi(2);
delta_3 = chi(3);
delta_4 = chi(4);
delta_5 = chi(5);
delta_6 = chi(6);
delta_7 = chi(7);
delta_8 = chi(8);
delta_9 = chi(9);
delta_10 = chi(10);
delta_11 = chi(11);

% Definición de matrices del sistema
M = [delta_1, 0, 0;
     0, delta_2, delta_3;
     0, delta_3, delta_4];

D = [delta_8, 0, 0;
     0, delta_9, delta_10;
     0, delta_10, delta_11];

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
    C = [0, -delta_5 * r(k), - delta_6 * v(k)-delta_3*r(k);
         delta_5*r(k), 0, delta_7 * u(k);
         delta_6 * v(k)+delta_3*r(k), -delta_7 * u(k), 0];
     
    % Cálculo de las derivadas de velocidad usando: M * d_vel = T - C*vel - D*vel
    d_vel = IM*([T_u(k); T_v(k); T_r(k)] - C * vel - D * vel);
    
    % Integración de Euler para obtener las velocidades en el siguiente instante
    u(k + 1) = u(k) + d_vel(1) * ts;
    % if u(k + 1)>2
    %     u(k+1)=2;
    % elseif u(k + 1)<-2
    %     u(k+1)=-2;
    % end
    u(k + 1) = saturacion_tanh(u(k + 1));
    v(k + 1) = v(k) + d_vel(2) * ts;
    r(k + 1) = r(k) + d_vel(3) * ts;
end

% figure;
% plot(t, T_u, 'r', 'DisplayName', 'T u');
% hold on;
% plot(t, T_r, 'b', 'DisplayName', 'T r');
% xlabel('Tiempo (s)');
% ylabel('Torques');
% title('Torques de Entrada');
% legend;
% grid on;


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


% Ajuste de espacio entre subplots
sgtitle('Resultados de Velocidades') % Título general para todos los subplots
error_u=calcularECM(u,vel_u);
error_v=calcularECM(v,vel_v);
error_r=calcularECM(r,vel_r);

% Imprime los resultados en la consola
fprintf('Error cuadrático medio de u: %.4f\n', error_u);
fprintf('Error cuadrático medio de v: %.4f\n', error_v);
fprintf('Error cuadrático medio de r: %.4f\n', error_r);

function ecm = calcularECM(vector1, vector2)
    % Comprueba que los vectores tengan el mismo tamaño
    if length(vector1) ~= length(vector2)
        error('Los vectores deben tener el mismo tamaño');
    end
    
    % Calcula el error cuadrático medio
    ecm = mean((vector1 - vector2).^2);
end

function salida = saturacion_tanh(valor)
    % valor=valor*10000;
    limite = 2;
    factor_suavidad = 6.4;
    salida = limite*factor_suavidad * tanh(valor / (limite*factor_suavidad));
    % salida= salida/10000;
end