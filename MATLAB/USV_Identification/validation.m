clc, clear all, close all;
clear tf;

    % Inicia la conexión ROS
    rosinit;
    duracion=60;
    ts=1/50;
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

    torque_commands = zeros(floor(duracion / ts), 5);  % [torque_left1, torque_left, torque_right1, torque_right]


    for k = 1:floor(duracion / ts)
        tic;
        % Genera señales usando seno y coseno
        t_current = k * ts;  % Tiempo actual
        torque_left = sin(2 * pi * (1/120) * t_current);    % Coseno para el propulsor izquierdo
        torque_right = sin(2 * pi * (1/120) * t_current);    % Coseno para el propulsor derecho

        % Publicar comandos de torque
        msg_left1 = rosmessage(pub_left1);
        msg_left = rosmessage(pub_left);
        msg_right1 = rosmessage(pub_right1);
        msg_right = rosmessage(pub_right);

        msg_left1.Data = torque_left;
        msg_left.Data = torque_left;
        msg_right1.Data = torque_right;
        msg_right.Data = torque_right;

        send(pub_left1, msg_left1);
        send(pub_left, msg_left);
        send(pub_right1, msg_right1);
        send(pub_right, msg_right);

        % Almacenar comandos de torque
        torque_commands(k, :) = [torque_left, torque_right,velocities_measured(1), velocities_measured(2),velocities_measured(3)];
        
        

        ta=toc;
        pause(ts-ta)
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
    T_right_thruster = calcularFuerzaPropulsor(torque_commands(i,1)) * 2;
    T_left_thruster = calcularFuerzaPropulsor(torque_commands(i,2)) * 2;

    % Calcula las fuerzas de traslación y rotación
    T_u(i) = T_left_thruster + T_right_thruster;  % Fuerza de traslación
    T_r(i) = d * (T_left_thruster - T_right_thruster);  % Fuerza de rotación
     % No hay fuerza lateral en este caso
end
T_v=T_u*0;

    % Graficar velocidades medidas
    figure;
    subplot(2, 1, 1);
    plot(t, torque_commands(:, 3), 'r', 'DisplayName', 'Velocidad u');
    hold on;
    plot(t, torque_commands(:, 4), 'g', 'DisplayName', 'Velocidad v');
    plot(t, torque_commands(:, 5), 'b', 'DisplayName', 'Velocidad r');
    xlabel('Tiempo (s)');
    ylabel('Velocidades (m/s o rad/s)');
    title('Velocidades medidas del USV');
    legend;
    grid on;

    % Graficar comandos de torque
    subplot(2, 1, 2);
    plot(t, T_u(:), 'r', 'DisplayName', 'Torque Izquierdo ');
    hold on;
    plot(t, T_r(:), 'm', 'DisplayName', 'Torque Derecho');
    xlabel('Tiempo (s)');
    ylabel('Torque (N*m)');
    title('Comandos de Torque');
    legend;
    grid on;
    
    vel_u=torque_commands(:, 3);
    vel_v=torque_commands(:, 4);
    vel_r=torque_commands(:, 5);

    save('sim_val.mat', 'T_u', 'T_r', 't', 'vel_u', 'vel_v', 'vel_r','ts');
    
    function odomCallback(~, msg)
        global velocities_measured;  % Declarar la variable global
        % Función callback para actualizar datos de odometría
        velocities_measured = [msg.Twist.Twist.Linear.X, msg.Twist.Twist.Linear.Y, msg.Twist.Twist.Angular.Z];
    end

