%% Clear variables and load data
clc;
clear all;
close all;
clear tf;

% Cargar los datos desde el archivo .mat
load('ident_usv.mat')

% Inicialización de las variables
up = zeros(1, length(t));
vp = zeros(1, length(t));
rp = zeros(1, length(t));

% Calcular las aceleraciones (derivadas de las velocidades)
for k = 1:length(t)
    if k > 1 
        up(k) = (vel_u(k) - vel_u(k-1)) / ts;
        vp(k) = (vel_v(k) - vel_v(k-1)) / ts;
        rp(k) = (vel_r(k) - vel_r(k-1)) / ts;
    else
        up(k) = 0;   
        vp(k) = 0; 
        rp(k) = 0; 
    end
end

% Parámetro de filtro
landa = 100000; % Lambda
F1 = tf(landa, [1 landa]);

% Filtrar las señales
u_f = lsim(F1, vel_u, t)';
v_f = lsim(F1, vel_v, t)';
r_f = lsim(F1, vel_r, t)';
up_f = lsim(F1, up, t)';
vp_f = lsim(F1, vp, t)';
rp_f = lsim(F1, rp, t)';

T_u_f = lsim(F1, T_u, t)';
T_r_f = lsim(F1, T_r, t)';

% Crear las matrices de datos para entrenamiento
nup = [up; vp; rp];
nu = [vel_u; vel_v; vel_r];
t_ref = [T_u; T_v ; T_r];
T_v = T_u_f * 0;
nu_f = [u_f; v_f; r_f];
nup_f = [up_f; vp_f; rp_f];
t_ref_f = [T_u_f; T_v; T_r_f];

% Matrices de regresores y torques
Y = [];
vef = [];

for k = 1:length(t)
    % Velocidades y aceleraciones actuales
    vel_u = nu_f(1, k);       % Velocidad en la dirección u
    vel_v = nu_f(2, k);       % Velocidad en la dirección v
    yaw_rate = nu_f(3, k);    % Velocidad angular (yaw rate)
    
    acc_u = nup_f(1, k);      % Aceleración en la dirección u
    acc_v = nup_f(2, k);      % Aceleración en la dirección v
    acc_r = nup_f(3, k);      % Aceleración angular (yaw)

    % Matriz regresora para el estado actual
    Yn = [
        acc_u,   0, -yaw_rate*yaw_rate, 0, -vel_v*yaw_rate,   -vel_v*yaw_rate,   0,      vel_u,      0,      0, 0;
        0,    acc_v,  acc_r,0 , vel_u*yaw_rate,  0,     vel_u*yaw_rate,      0,     vel_v,    yaw_rate,  0;
        0,0, acc_v+vel_u*yaw_rate, acc_r , 0,   vel_v*vel_u,  -vel_v*vel_u,      0,      0,   vel_v,   yaw_rate
    ];

    % Se acumula la matriz regresora y los torques
    Y = [Y; Yn];
    vef = [vef; t_ref_f(:, k)]; % Aquí T_f contiene las fuerzas y momentos [T_u; T_v; T_r] en el tiempo k
end

% Calcula los parámetros usando mínimos cuadrados
delta = pinv(Y) * vef;
save('delta_valores_4.mat', 'delta');

% Imprime los valores de los parámetros identificados
disp('Parámetros identificados (delta):');
for i = 1:length(delta)
    fprintf('delta_%d es %.4f\n', i, delta(i));
end

%% Entrenamiento de la Red Neuronal para mejorar el modelo

% Definir las entradas y salidas de la red neuronal
% Entradas: concatenación de las velocidades y aceleraciones (nu_f y nup_f)
inputs = [nu_f; nup_f]'; % Cambié esto para que las entradas sean de tamaño (num_samples, num_features)
% Salidas: error entre el modelo actual y las fuerzas y momentos reales
outputs = vef - Y * delta; % El tamaño de las salidas debe coincidir con el número de muestras

% Verificar que las dimensiones coinciden
disp(size(inputs));   % Debe ser (num_samples, num_features)
disp(size(outputs));  % Debe ser (num_samples, 3) o el tamaño de vef

% Normalizar las entradas
inputs = (inputs - mean(inputs)) ./ std(inputs); % Normalización Z-score

% Crear la red neuronal
hiddenLayerSize = 10; % Número de neuronas en la capa oculta (ajustable)
net = feedforwardnet(hiddenLayerSize);

% Configurar la red para regresión
net.performFcn = 'mse'; % Error cuadrático medio
net.trainFcn = 'trainlm'; % Algoritmo de entrenamiento Levenberg-Marquardt

% Dividir los datos en entrenamiento, validación y prueba
[net, tr] = train(net, inputs', outputs');

% Realizar predicciones con la red entrenada
predictions = net(inputs');

% Incorporar las predicciones de la red neuronal en el modelo
N = reshape(predictions, size(vef)); % Ajustar la forma si es necesario
new_vef = vef + N; % Se ajusta el vector de fuerzas y momentos con la corrección

% Recalcular la matriz regresora con la corrección
Y_new = [];
for k = 1:length(t)
    Yn_new = [
        acc_u,   0, -yaw_rate*yaw_rate, 0, -vel_v*yaw_rate,   -vel_v*yaw_rate,   0,      vel_u,      0,      0, 0;
        0,    acc_v,  acc_r,0 , vel_u*yaw_rate,  0,     vel_u*yaw_rate,      0,     vel_v,    yaw_rate,  0;
        0,0, acc_v+vel_u*yaw_rate, acc_r , 0,   vel_v*vel_u,  -vel_v*vel_u,      0,      0,   vel_v,   yaw_rate
    ];
    Y_new = [Y_new; Yn];    
end

% Actualiza los parámetros identificados usando la corrección
delta_new = pinv(Y_new) * new_vef;
save('delta_valores_ajustados.mat', 'delta_new');

% Imprime los nuevos parámetros identificados
disp('Parámetros ajustados (delta_new):');
for i = 1:length(delta_new)
    fprintf('delta_%d es %.4f\n', i, delta_new(i));
end
