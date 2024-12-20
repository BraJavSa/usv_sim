% Definición del tiempo total y el tiempo de muestreo
clear all;
t_total = 90;              % Duración total del experimento en segundos
fs = 50;                   % Frecuencia de muestreo en Hz

% Función para normalizar cada señal al rango [-1, 1]
normalize_signal = @(signal) 2 * (signal - min(signal)) / (max(signal) - min(signal)) - 1;

% Número total de muestras
total_samples = t_total * fs;
funtions = 19;
% Inicialización de las señales
propulsor1 = cell(1, funtions); % Almacenará funciones para el primer propulsor
propulsor2 = cell(1, funtions); % Almacenará funciones para el segundo propulsor
t = 0:1/fs:t_total/funtions; % Vector de tiempo (en segundos)

% Funciones de la forma seno y coseno con frecuencias ajustadas para 90 segundos
propulsor1{1} = normalize_signal(sin(3 * t));   
propulsor2{1} = normalize_signal(cos(3 * t));   

propulsor1{2} = normalize_signal( sin(2 * t)) * 0.8; 
propulsor2{2} = normalize_signal( cos(2* t)) * 0.8; 

propulsor1{3} = normalize_signal(0.5 * sin(4 * t)); 
propulsor2{3} = normalize_signal(0.5 * sin(4* t)); 

propulsor1{4} = normalize_signal(square(2 * t));      
propulsor2{4} = normalize_signal(-square(2 * t));     

propulsor1{5} = normalize_signal(square(5 * t)) * 0.5;      
propulsor2{5} = normalize_signal(-square(5 * t)) * 0.5;     
   

% Funciones con pasos abruptos
half_length = floor(length(t)/2); % Tamaño entero para dividir en mitades
propulsor1{6} = normalize_signal([ones(1, half_length), -ones(1, half_length)]); 
propulsor2{6} = normalize_signal([-ones(1, half_length), ones(1, half_length)]) * 0.8; 

quarter_length = floor(length(t)/4); % Tamaño entero para dividir en cuartos
propulsor1{7} = normalize_signal([ones(1, quarter_length), -ones(1, quarter_length), ...
                                  ones(1, quarter_length), -ones(1, quarter_length)]) * 0.8; 
propulsor2{7} = normalize_signal([ones(1, quarter_length), -ones(1, quarter_length), ...
                                  ones(1, quarter_length), -ones(1, quarter_length)]) * -0.8; 

% Funciones aleatorias entre -1 y 1
rng(0); 

propulsor1{8} = -ones(1, length(t)); 
propulsor2{8} = -ones(1, length(t)); 

propulsor1{9} = 0.5 * ones(1, length(t)); 
propulsor2{9} = 0.5 * ones(1, length(t)); 

propulsor1{10} = -0.8 * ones(1, length(t)); 
propulsor2{10} = -0.8 * ones(1, length(t)); 

propulsor1{11} = ones(1, length(t)); 
propulsor2{11} = zeros(1, length(t)); 

propulsor1{12} = zeros(1, length(t)); 
propulsor2{12} = ones(1, length(t)); 

propulsor1{13} = ones(1, length(t)); 
propulsor2{13} = -ones(1, length(t)); 

propulsor1{14} = -ones(1, length(t)); 
propulsor2{14} = ones(1, length(t)); 

propulsor1{15} = normalize_signal(0.8 * sin(0.2 * t) + cos(0.2 * t)) * 0.8; 
propulsor2{15} = normalize_signal(0.8 * cos(0.2 * t) + sin(0.2 * t)) * 0.8; 

propulsor1{16} = normalize_signal(sin( t)) * 0.9; 
propulsor2{16} = normalize_signal(sin(t)) * 0.9; 

propulsor1{17} = normalize_signal(0.8 * cos(t)) * 0.5; 
propulsor2{17} = normalize_signal(0.8 * cos(t)) * 0.5;
propulsor1{18} = ones(1, length(t)); 
propulsor2{18} = ones(1, length(t)); 
propulsor1{19} = -ones(1, length(t)); 
propulsor2{19} = -ones(1, length(t)); 
% Inicialización de vectores acumulativos para ambos propulsores
cmd_r = []; % Vector para almacenar todos los valores del propulsor 1
cmd_l = []; % Vector para almacenar todos los valores del propulsor 2

% Concatenación de las señales generadas
for i = 1:funtions
    cmd_r = [cmd_r , propulsor1{i}]; % Apilando propulsor 1
    cmd_l = [cmd_l, propulsor2{i}]; % Apilando propulsor 2
end

% Asegurarse de que las señales tengan exactamente 4500 muestras
cmd_r = cmd_r(1:total_samples);
cmd_l = cmd_l(1:total_samples);

% Graficar las señales generadas
figure;
subplot(2,1,1);
plot((0:length(cmd_r)-1)/fs, cmd_r, 'r');
xlabel('Time (s)');
ylabel('Value');
title('Signal for Right Propeller');
grid on;

subplot(2,1,2);
plot((0:length(cmd_l)-1)/fs, cmd_l, 'b');
xlabel('Time (s)');
ylabel('Value');
title('Signal for Left Propeller');
grid on;

% Guardar las señales generadas en un archivo
save('propulsor_signals2.mat', 'cmd_r', 'cmd_l');
