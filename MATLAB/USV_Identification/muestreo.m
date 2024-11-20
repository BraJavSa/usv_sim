clc;
clear all;
close all;
clear tf;

% Inicia la conexión ROS
rosinit;
load('propulsor_signals2.mat', 'cmd_l', 'cmd_r');

ts = 1 / 50;  % Intervalo de tiempo
duracion = length(cmd_r) * ts;

% Inicializa variables para almacenar datos
global velocities_measured;  % Declarar la variable global
velocities_measured = zeros(1, 3);  % [u, v, r]

% Publicadores para los torques de los propulsores
pub_signals = rospublisher('/wamv/signals', 'std_msgs/Float32MultiArray');  % Publicador para señales

% Crea un subscriber para el tópico de odometría
sub_odometry = rossubscriber('/boat/odom', @odomCallback);

tau = zeros(floor(duracion / ts), 2);  % [torque_left, torque_right]

nu = zeros(floor(duracion / ts), 3);  % [u, v, r]

for k = 1:floor(duracion / ts - 1)
    tic;

    torque_left = cmd_l(k);  % Torque para el propulsor izquierdo
    torque_right = cmd_r(k);  % Torque para el propulsor derecho
    
    % Crear un mensaje Float32MultiArray con los torques izquierdo y derecho
    msg = rosmessage('std_msgs/Float32MultiArray');
    msg.Data = [torque_left, torque_right];  % Asigna los torques izquierdo y derecho al mensaje

    % Publica el mensaje
    send(pub_signals, msg);

    % Almacenar comandos de torque
    tau(k, :) = [torque_left, torque_right];

    ta = toc;
    pause(ts - ta);  % Mantener la tasa de muestreo
    nu(k + 1, :) = [velocities_measured(1), velocities_measured(2), velocities_measured(3)];
end
msg.Data = [0, 0];  % Asigna los torques izquierdo y derecho al mensaje

 % Publica el mensaje
send(pub_signals, msg);


% Cerrar la conexión ROS
rosshutdown;

% Análisis y comparación de datos
t = 0:ts:duracion - ts;  % Vector de tiempo

% Define la distancia entre los propulsores
d = 1.4;

T_u = zeros(floor(duracion / ts), 1);
T_r = zeros(floor(duracion / ts), 1);

% Aplica la función calcularFuerzaPropulsor para obtener los torques en cada paso
for i = 1:floor(duracion / ts)
    % Calcula las fuerzas de cada propulsor
    T_right_thruster = calcularFuerzaPropulsor(tau(i, 1)) * 2;
    T_left_thruster = calcularFuerzaPropulsor(tau(i, 2)) * 2;

    % Calcula las fuerzas de traslación y rotación
    T_u(i) = T_left_thruster + T_right_thruster;  % Fuerza de traslación
    T_r(i) = d * (T_left_thruster - T_right_thruster);  % Fuerza de rotación
    % No hay fuerza lateral en este caso
end
T_v = T_u * 0;

vel_u = nu(:, 1);
vel_v = nu(:, 2);
vel_r = nu(:, 3);

save('ident_usv_3.mat', 'T_u', 'T_r', 'T_v', 't', 'vel_u', 'vel_v', 'vel_r', 'ts');
disp("Muestreo terminado");

function odomCallback(~, msg)
    global velocities_measured;  % Declarar la variable global
    % Función callback para actualizar datos de odometría
    velocities_measured = [msg.Twist.Twist.Linear.X, msg.Twist.Twist.Linear.Y, msg.Twist.Twist.Angular.Z];
end
