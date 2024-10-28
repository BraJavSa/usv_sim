clc, clear all, close all;
clear tf;

    % Inicia la conexión ROS
    rosinit;
    load('propulsor_signals.mat', 'cmd_l', 'cmd_r');
    
    ts=1/50;
    duracion=length(cmd_r)*ts;
    
    % Publicadores para los torques de los propulsores
    pub_left1 = rospublisher('/wamv/thrusters/left1_thrust_cmd', 'std_msgs/Float32');
    pub_left = rospublisher('/wamv/thrusters/left_thrust_cmd', 'std_msgs/Float32');
    pub_right1 = rospublisher('/wamv/thrusters/right1_thrust_cmd', 'std_msgs/Float32');
    pub_right = rospublisher('/wamv/thrusters/right_thrust_cmd', 'std_msgs/Float32');

    stop_pub = rospublisher('/stop_logging', 'std_msgs/Bool');

    
    for k = 1:floor(duracion / ts-1)
        tic;
              
        torque_left = cmd_l(k);    % Coseno para el propulsor izquierdo
        torque_right = cmd_r(k);    % Coseno para el propulsor derecho
        
        sendTorques(torque_left, torque_right, pub_left1, pub_left, pub_right1, pub_right);

        % Almacenar comandos de torque
        
        
        ta=toc;
        pause(ts-ta);
        
       
    end
% Crear el mensaje de tipo Bool
stop_msg = rosmessage(stop_pub);

% Establecer el valor de data en true
stop_msg.Data = true;

% Publicar el mensaje en el tópico
send(stop_pub, stop_msg);
    % Cerrar la conexión ROS
rosshutdown;
    
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


