% Ignorar advertencias específicas
warning('off', 'all'); % Desactiva todas las advertencias
warning('on', 'MATLAB:matlab:dispatcher:UseOfDeprecatedFunction'); % Reactiva algunas advertencias si lo deseas

% Cargar los datos
load('ident_usv.mat');


% Rango de tiempo
t_vector = t;
dt = ts;

% Velocidades
velocities = [vel_u, vel_v, vel_r];

% Torques aplicados
torques = [T_u, T_v, T_r];

u = torques;

% Entrenamiento
u_train = u;

% Inicializa el conjunto de entrenamiento múltiple
u_train_multi = repmat({u_train'}, 1, 4); % Replicar u_train 4 veces

X = velocities; % Variables independientes (velocidades)
Y = torques; % Variables dependientes (torques)

% Ajuste de un modelo polinómico (o algún otro modelo de interés)
degree = 1;
p_u = polyfit(X(:, 1), Y(:, 1), degree);
p_v = polyfit(X(:, 2), Y(:, 2), degree);
p_r = polyfit(X(:, 3), Y(:, 3), degree);

% Predecir los valores utilizando el modelo ajustado
y_pred_u = polyval(p_u, X(:, 1));
y_pred_v = polyval(p_v, X(:, 2));
y_pred_r = polyval(p_r, X(:, 3));

% Crear figura
figure;

% Subplot 1: Velocidad u (real)
subplot(2, 3, 1);
plot(1:length(vel_u), vel_u, 'r', 'DisplayName', 'Modelo Real');
xlabel('Time');
ylabel('Velocidad u');
title('Datos reales (u)');
legend;
grid on;

% Subplot 2: Velocidad v (real)
subplot(2, 3, 2);
plot(1:length(vel_v), vel_v, 'r', 'DisplayName', 'Modelo Real');
xlabel('Time');
ylabel('Velocidad v');
title('Datos reales (v)');
legend;
grid on;

% Subplot 3: Velocidad r (real)
subplot(2, 3, 3);
plot(1:length(vel_r), vel_r, 'r', 'DisplayName', 'Modelo Real');
xlabel('Time');
ylabel('Velocidad r');
title('Datos reales (r)');
legend;
grid on;

% Subplot 4: Predicción u
subplot(2, 3, 4);
plot(1:length(y_pred_u), y_pred_u, 'b', 'DisplayName', 'Predicción');
xlabel('Time');
ylabel('Predicción u');
title('Predicción (u)');
legend;
grid on;

% Subplot 5: Predicción v
subplot(2, 3, 5);
plot(1:length(y_pred_v), y_pred_v, 'b', 'DisplayName', 'Predicción');
xlabel('Time');
ylabel('Predicción v');
title('Predicción (v)');
legend;
grid on;

% Subplot 6: Predicción r
subplot(2, 3, 6);
plot(1:length(y_pred_r), y_pred_r, 'b', 'DisplayName', 'Predicción');
xlabel('Time');
ylabel('Predicción r');
title('Predicción (r)');
legend;
grid on;

% Ajuste del tamaño de la figura
set(gcf, 'Position', [100, 100, 900, 600]);  % Ajusta el tamaño de la figura