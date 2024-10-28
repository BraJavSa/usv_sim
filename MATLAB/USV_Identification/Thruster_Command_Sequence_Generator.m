% Definición del tiempo total y el tiempo de muestreo
t_total = 10;              % Duración de cada función en segundos
fs = 50;                   % Frecuencia de muestreo en Hz
t = linspace(0, t_total, t_total * fs); % Vector de tiempo

% Función para normalizar cada señal al rango [-1, 1]
normalize_signal = @(signal) 2 * (signal - min(signal)) / (max(signal) - min(signal)) - 1;

% Generación y normalización de las funciones
propulsor1 = cell(1, 23); % Almacenará funciones para el primer propulsor
propulsor2 = cell(1, 23); % Almacenará funciones para el segundo propulsor

% Funciones de la forma seno y coseno
propulsor1{1} = normalize_signal(sin(0.5 * t));   
propulsor2{1} = normalize_signal(cos(0.5 * t));   

propulsor1{2} = normalize_signal(0.8 * sin(2 * t))*0.8; 
propulsor2{2} = normalize_signal(0.8 * cos(2 * t))*0.8; 

propulsor1{3} = normalize_signal(0.5 * sin(4 * t)); 
propulsor2{3} = normalize_signal(0.5 * cos(4 * t)); 

% Funciones con cambios abruptos
propulsor1{4} = normalize_signal(0.3 * sin(9 * t)); 
propulsor2{4} = normalize_signal(0.3 * cos(49 * t));    

propulsor1{5} = normalize_signal(square(2 * t));      
propulsor2{5} = normalize_signal(-square(2 * t));     

propulsor1{6} = normalize_signal(square(4 * t)) * 0.5;      
propulsor2{6} = normalize_signal(-square(4 * t)) * 0.5;     

% Funciones mixtas con seno, coseno y cambios abruptos
propulsor1{7} = normalize_signal(sin(0.2 * t) + square(0.5 * t) * 0.3); 
propulsor2{7} = normalize_signal(cos(0.2 * t) - square(0.5 * t) * 0.3); 

propulsor1{8} = normalize_signal(sin(3 * t) + 0.4 * square(0.5 * t));   
propulsor2{8} = normalize_signal(cos(3 * t) - 0.4 * square(0.5 * t));   

% Funciones con pasos abruptos
propulsor1{9} = normalize_signal([ones(1, length(t)/2), -ones(1, length(t)/2)]); 
propulsor2{9} = normalize_signal([-ones(1, length(t)/2), ones(1, length(t)/2)]) * 0.8; 

propulsor1{10} = normalize_signal([ones(1, length(t)/4), -ones(1, length(t)/4), ones(1, length(t)/4), -ones(1, length(t)/4)]) * 0.8; 
propulsor2{10} = normalize_signal([ones(1, length(t)/4), -ones(1, length(t)/4), ones(1, length(t)/4), -ones(1, length(t)/4)]) * -0.8; 

% Funciones aleatorias entre -1 y 1
rng(0); 
propulsor1{11} = normalize_signal(2 * rand(1, length(t)) - 1); 
propulsor2{11} = normalize_signal(2 * rand(1, length(t)) - 1); 

propulsor1{12} = normalize_signal(sin(0.3 * t) + 0.3 * (2 * rand(1, length(t)) - 1)); 
propulsor2{12} = normalize_signal(cos(0.3 * t) + 0.3 * (2 * rand(1, length(t)) - 1)); 

% Nuevas señales constantes
propulsor1{13} = ones(1, length(t)); 
propulsor2{13} = ones(1, length(t)); 

propulsor1{14} = -ones(1, length(t)); 
propulsor2{14} = -ones(1, length(t)); 

propulsor1{15} = 0.5 * ones(1, length(t)); 
propulsor2{15} = 0.5 * ones(1, length(t)); 

propulsor1{16} = -0.8 * ones(1, length(t)); 
propulsor2{16} = -0.8 * ones(1, length(t)); 

propulsor1{17} = ones(1, length(t)); 
propulsor2{17} = zeros(1, length(t)); 

propulsor1{18} = zeros(1, length(t)); 
propulsor2{18} = ones(1, length(t)); 

propulsor1{19} = ones(1, length(t)); 
propulsor2{19} = -ones(1, length(t)); 

propulsor1{20} = -ones(1, length(t)); 
propulsor2{20} = ones(1, length(t)); 


propulsor1{21} = normalize_signal(0.8 * sin(2 * t)+cos(2 * t))*0.8; 
propulsor2{21} = normalize_signal(0.8 * cos(2 * t)+sin(2 * t))*0.8; 

propulsor1{22} = normalize_signal(0.8 * sin(2 * t))*0.9; 
propulsor2{22} = normalize_signal(0.8 * sin(2 * t))*0.9; 


propulsor1{23} = normalize_signal(0.8 * cos(2 * t))*0.5; 
propulsor2{23} = normalize_signal(0.8 * cos(2 * t))*0.5; 
% Inicialización de vectores acumulativos para ambos propulsor
cmd_r = []; % Vector para almacenar todos los valores del propulsor 1
cmd_l = []; % Vector para almacenar todos los valores del propulsor 2

% Concatenación de cada función en los vectores acumulativos
for i = 1:23
    cmd_r  = [cmd_r , propulsor1{i}]; % Apilando propulsor 1
    cmd_l = [cmd_l, propulsor2{i}]; % Apilando propulsor 2
end


% Gráfica de los vectores acumulativos
figure;
subplot(2,1,1);
plot((0:length(cmd_r)-1) / fs, cmd_r, 'r');
xlabel('Tiempo (s)');
ylabel('Propulsor 1');
title('CMD de excitacion de Propulsor derecho');
grid on;

subplot(2,1,2);
plot((0:length(cmd_l)-1) / fs, cmd_l, 'b');
xlabel('Tiempo (s)');
ylabel('Propulsor 2');
title('CMD de excitacion de Propulsor izquierdo');
grid on;

% Guardar las señales generadas en un archivo .mat
save('propulsor_signals.mat', 'cmd_r', 'cmd_l');
