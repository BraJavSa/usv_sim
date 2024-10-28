clc, clear all, close all;
clear tf;

    % Inicia la conexión ROS
    rosinit;
    
    ts=1/50;
    duracion=30;
    total=30/ts;
        % Inicializa variables para almacenar datos
    global velocities_measured;  % Declarar la variable global
    global left_thrust right_thrust


    velocities_measured = zeros(1, 3);  % [u, v, r]
    left_thrust = 0;
    right_thrust = 0;
    
    % Crear los suscriptores y asignar los callbacks
    sub_left = rossubscriber('/wamv/thrusters/left_thrust_cmd', 'std_msgs/Float32', @leftThrustCallback);
    sub_right = rossubscriber('/wamv/thrusters/right_thrust_cmd', 'std_msgs/Float32', @rightThrustCallback);
    sub_odometry = rossubscriber('/boat/odom', @odomCallback);
    tau = zeros(floor(duracion / ts),2);  % [torque_left1, torque_left, torque_right1, torque_right]

    nu = zeros(floor(duracion / ts),3);  % [torque_left1, torque_left, torque_right1, torque_right]


    for k = 1:floor(duracion / ts-1)
        tic;
              
               
        tau(k, :) = [left_thrust, right_thrust];
        ta=toc;
        pause(ts-ta);
        nu(k+1, :) = [velocities_measured(1), velocities_measured(2),velocities_measured(3)];       
    end

    % Cerrar la conexión ROS
rosshutdown
disp("Muestreo terminado")
    % Análisis y comparación de datos
    t = 0:ts:duracion-ts; % Vector de tiempo


    % Define la distancia entre los propulsores
d = 1.4;

T_u = zeros(floor(duracion / ts), 1);
T_r = zeros(floor(duracion / ts), 1);


% Aplica la función calcularFuerzaPropulsor para obtener los torques en cada paso
for i = 1:floor(duracion / ts)

    % Calcula las fuerzas de cada propulsor
    T_right_thruster = calcularFuerzaPropulsor(tau(i,1)) * 2;
    T_left_thruster = calcularFuerzaPropulsor(tau(i,2)) * 2;

    % Calcula las fuerzas de traslación y rotación
    T_u(i) = T_left_thruster + T_right_thruster;  % Fuerza de traslación
    T_r(i) = d * (T_left_thruster - T_right_thruster);  % Fuerza de rotación
     % No hay fuerza lateral en este caso
end
T_v=T_u*0;

    
    
    vel_u=nu(:, 1);
    vel_v=nu(:, 2);
    vel_r=nu(:, 3);

    save('muestreo_externo.mat', 'T_u', 'T_r','T_v', 't', 'vel_u', 'vel_v', 'vel_r','ts');
    disp("Muestreo terminado")
    function odomCallback(~, msg)
        global velocities_measured;  % Declarar la variable global
        % Función callback para actualizar datos de odometría
        velocities_measured = [msg.Twist.Twist.Linear.X, msg.Twist.Twist.Linear.Y, msg.Twist.Twist.Angular.Z];
    end


% Callback para el propulsor izquierdo
function leftThrustCallback(~, msg)
    global left_thrust  % Declarar la variable como global dentro del callback
    left_thrust = msg.Data;  % Almacenar el valor recibido en la variable global
end

% Callback para el propulsor derecho
function rightThrustCallback(~, msg)
    global right_thrust  % Declarar la variable como global dentro del callback
    right_thrust = msg.Data;  % Almacenar el valor recibido en la variable global
end

