clc, clear all, close all
load('delta_valores4.mat', 'delta');

% Asignar los valores de delta a variables individuales
delta_1 = delta(1);
delta_2 = delta(2);
delta_3 = delta(3);
delta_4 = delta(4);
delta_5 = delta(5);
delta_6 = delta(6);
delta_7 = delta(7);
delta_8 = delta(8);
delta_9 = delta(9);
delta_10 = delta(10);
delta_11 = delta(11);

global right_thrust left_thrust
right_thrust = 0;  % Inicializar a 0
left_thrust = 0;   % Inicializar a 0
ts = 1/50;  % Tiempo de muestreo
total = 20;  % Duración total de la simulación en segundos
t = 0:ts:total;  % Tiempo total de simulación

% Definición de matrices del sistema
M = [delta_1, 0, 0;
     0, delta_2, delta_3;
     0, delta_3, delta_4];

D = [delta_8, 0, 0;
     0, delta_9, delta_10;
     0, delta_10, delta_11];

% Condiciones iniciales de velocidad
u = zeros(length(t), 1);
v = zeros(length(t), 1);
r = zeros(length(t), 1);
T_u = zeros(length(t), 1);
T_v = zeros(length(t), 1);
% Torques de entrada (asumimos T_v = 0, ya que solo están T_u y T_r)
T_v = 0;
IM = inv(M);

% Inicializar el nodo de ROS
rosinit;
node = ros.Node('/matlab_global_node');  % Crear el nodo

% Crear suscriptores para los comandos de empuje
sub_left = rossubscriber( '/wamv/thrusters/left_thrust_cmd', 'std_msgs/Float32', @leftThrustCallback);
sub_right = rossubscriber( '/wamv/thrusters/right_thrust_cmd', 'std_msgs/Float32', @rightThrustCallback);
posePub = rospublisher('/usv2/pose', 'geometry_msgs/Pose');
newPose = rosmessage(posePub);


x(1) = 0;  % Posición inicial en x
y(1) = 0;  % Posición inicial en y
theta(1) = 0;  % Orientación inicial

for k = 1:length(t) - 1
    % Estado de velocidades actuales
    vel = [u(k); v(k); r(k)];
    T_u(k) = left_thrust+right_thrust;
    T_r(k) = 1.4*(right_thrust-left_thrust);
    
    % Cálculo del vector C(vel) * vel
    C = [0, -delta_5 * r(k), -delta_6 * v(k) - delta_3 * r(k);
         delta_5 * r(k), 0, delta_7 * u(k);
         delta_6 * v(k) + delta_3 * r(k), -delta_7 * u(k), 0];

    % Cálculo de las derivadas de velocidad
    d_vel = IM * ([T_u(k); T_v; T_r(k)] - C * vel - D * vel);

    % Integración de Euler
    u(k + 1) = saturacion_tanh(u(k) + d_vel(1) * ts);
    v(k + 1) = v(k) + d_vel(2) * ts;
    r(k + 1) = r(k) + d_vel(3) * ts;
    
    % Actualizar posición y orientación
    theta(k + 1) = theta(k) + r(k) * ts;
    theta(k + 1) = theta(k + 1);
    x(k + 1) = x(k) + u(k) * cos(theta(k)) * ts - v(k) * sin(theta(k)) * ts;
    y(k + 1) = y(k) + u(k) * sin(theta(k)) * ts + v(k) * cos(theta(k)) * ts;

    % Preparar el mensaje para publicar
    newPose.Position.X = x(k + 1);
    newPose.Position.Y = y(k + 1);
    newPose.Position.Z = 0;  % Mantener Z constante

    % Convertir theta a cuaterniones
    quat = eul2quat([0, 0, theta(k + 1)]);
    newPose.Orientation.X = quat(4);
    newPose.Orientation.Y = quat(3);
    newPose.Orientation.Z = quat(2);
    newPose.Orientation.W = quat(1);

    % Publicar la nueva posición
    send(posePub, newPose);
    pause(0.012);
end

rosshutdown;

% Callback para el propulsor izquierdo
function leftThrustCallback(~, msg)
    global left_thrust
    left_thrust = calcularFuerzaPropulsor(msg.Data);
end

% Callback para el propulsor derecho
function rightThrustCallback(~, msg)
    global right_thrust
    right_thrust = calcularFuerzaPropulsor(msg.Data);
end

function salida = saturacion_tanh(valor)
    limite = 2;
    factor_suavidad = 6.4;
    salida = limite * factor_suavidad * tanh(valor / (limite * factor_suavidad));
end


