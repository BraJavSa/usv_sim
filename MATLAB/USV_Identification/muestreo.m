clc, clear all, close all;
clear tf;

    % Inicia la conexión ROS
    rosinit;
    load('propulsor_signals.mat', 'cmd_l', 'cmd_r');
    
    ts=1/50;
    duracion=length(cmd_r)*ts;
        % Inicializa variables para almacenar datos
    global velocities_measured;  % Declarar la variable global

    velocities_measured = zeros(1, 3);  % [u, v, r]

    % Publicadores para los torques de los propulsores
    pub_left1 = rospublisher('/wamv/thrusters/left1_thrust_cmd', 'std_msgs/Float32');
    pub_left = rospublisher('/wamv/thrusters/left_thrust_cmd', 'std_msgs/Float32');
    pub_right1 = rospublisher('/wamv/thrusters/right1_thrust_cmd', 'std_msgs/Float32');
    pub_right = rospublisher('/wamv/thrusters/right_thrust_cmd', 'std_msgs/Float32');

    % Crea un subscriber para el tópico de odometría
    sub_odometry = rossubscriber('/boat/odom', @odomCallback);
        tau = zeros(floor(duracion / ts),2);  % [torque_left1, torque_left, torque_right1, torque_right]

    nu = zeros(floor(duracion / ts),3);  % [torque_left1, torque_left, torque_right1, torque_right]


    for k = 1:floor(duracion / ts-1)
        tic;
              
        torque_left = cmd_l(k);    % Coseno para el propulsor izquierdo
        torque_right = cmd_r(k);    % Coseno para el propulsor derecho
        
        sendTorques(torque_left, torque_right, pub_left1, pub_left, pub_right1, pub_right);

        % Almacenar comandos de torque
        
        tau(k, :) = [torque_left, torque_right];
        ta=toc;
        pause(ts-ta);
        nu(k+1, :) = [velocities_measured(1), velocities_measured(2),velocities_measured(3)];
       
    end

    % Cerrar la conexión ROS
rosshutdown
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

    save('ident_usv.mat', 'T_u', 'T_r','T_v', 't', 'vel_u', 'vel_v', 'vel_r','ts');
    disp("Muestreo terminado")
    function odomCallback(~, msg)
        global velocities_measured;  % Declarar la variable global
        % Función callback para actualizar datos de odometría
        velocities_measured = [msg.Twist.Twist.Linear.X, msg.Twist.Twist.Linear.Y, msg.Twist.Twist.Angular.Z];
    end

    % Define la función para enviar comandos de torque
function sendTorques(torque_left, torque_right, pub_left1, pub_left, pub_right1, pub_right)
    % Crear y enviar mensajes de torque para los propulsores izquierdo y derecho
    
    % Mensajes de torque para el propulsor izquierdo
    msg_left1 = rosmessage(pub_left1);
    msg_left = rosmessage(pub_left);
    
    msg_left1.Data = torque_left;
    msg_left.Data = torque_left;
    
    send(pub_left1, msg_left1);
    send(pub_left, msg_left);
    
    % Mensajes de torque para el propulsor derecho
    msg_right1 = rosmessage(pub_right1);
    msg_right = rosmessage(pub_right);
   
    msg_right1.Data = torque_right;
    msg_right.Data = torque_right;
    
    send(pub_right1, msg_right1);
    send(pub_right, msg_right);
end


