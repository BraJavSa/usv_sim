%% IDENTIFICATION OF PARAMETERS USV DYNAMIC %%

%% clear variables
clc, clear all, close all;
clear tf;




%% LOAD VALUES FROM MATRICES
load('usv_data.mat')
desfase=11515; % minimo  de datos tomados

ts=1/50;
t = 0:ts:ts*desfase-ts;
N = length(t);


% REFERENCE SIGNALS FOR WEBOTS
cmd_right_thruster = right_thruster(1,1:desfase);
cmd_left_thruster = left_thruster(1,1:desfase);

T__right_thruster = zeros(size(cmd_right_thruster));
T__left_thruster = zeros(size(cmd_left_thruster));
% Recorre el vector cmd y aplica la función calcularFuerzaPropulsor
for i = 1:length(T__right_thruster)
    T__right_thruster(i) = calcularFuerzaPropulsor(cmd_right_thruster(i))*2; 
    T__left_thruster(i) = calcularFuerzaPropulsor(cmd_left_thruster(i))*2;  
end

% Define la distancia entre los propulsores
d = 1.4; 

% Calcula la fuerza de traslación
T_u = T__left_thruster + T__right_thruster;

% Calcula la fuerza de rotación
T_r = d * (T__left_thruster - T__right_thruster);
T_v = 0 * T__left_thruster ;


% REAL SYSTEM VELICITIES FOR WEBOTS DATA
u = double(odom_lin_vel_x(1,1:length(t)));
v = double(odom_lin_vel_y(1,1:length(t)));
r = double(odom_ang_vel_z(1,1:length(t)));
up = zeros(1, length(t));
vp = zeros(1, length(t));
rp = zeros(1, length(t));


% 2) ACELERATIONS System
for k=1:length(t)
    if k>1 
        up(k)=(u(k)- u(k-1))/ts;
        vp(k)=(v(k)- v(k-1))/ts;
        rp(k)=(r(k)- r(k-1))/ts;
    else
        up(k)=0;   
        vp(k)=0; 
        rp(k)=0; 
         
    end
end

%% ACELERATION SYSTEM
% ulp = [0 , diff(ul)/ts];
% ump = [0 , diff(um)/ts];
% unp = [0 , diff(un)/ts];
% wp = [0 , diff(w)/ts];

landa = 1;%lambda
F1=tf(landa,[1 landa]);

u_f=lsim(F1,u,t)';
v_f=lsim(F1,v,t)';
r_f=lsim(F1,r,t)';

up_f=lsim(F1,up,t)';
vp_f=lsim(F1,vp,t)';
rp_f=lsim(F1,rp,t)';


T_u_f=lsim(F1,T_u,t)';
T_r_f=lsim(F1,T_r,t)';


nup = [up; vp; rp];
nu = [u; v; r];
t_ref = [T_u; T_v ; T_r];

nu_f = [u_f; v_f; r_f];
nup_f = [up_f; vp_f; rp_f];
t_ref_f = [T_u_f; T_v; T_r_f];

a = 0;
b = 0;

L = [a;b];

Y=[];
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
        acc_u,   -vel_v*yaw_rate,   -yaw_rate^2,   0,      vel_u,      0,      0,      0;
        vel_u*yaw_rate,     acc_v,  acc_r,  0,      0,      vel_u,      yaw_rate,      0;
        -vel_u*vel_v,    vel_u*vel_v,    acc_v + vel_u*yaw_rate,    acc_r,  0,      0,      vel_v,      yaw_rate
    ];

    % Se acumula la matriz regresora y los torques
    Y = [Y; Yn];
    vef = [vef; t_ref_f(:, k)]; % Aquí T_f contiene las fuerzas y momentos [T_u; T_v; T_r] en el tiempo k
end

% Calcula los parámetros usando mínimos cuadrados
delta = pinv(Y) * vef;
save('delta_valores_1.mat', 'delta');

% Imprime los valores de los parámetros identificados con un formato personalizado
disp('Parámetros identificados (delta):');
for i = 1:length(delta)
    fprintf('delta_%d es %.4f\n', i, delta(i));
end


