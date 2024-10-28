%% clear variables
clc, clear all, close all;
clear tf;




%% LOAD VALUES FROM MATRICES
load('ident_usv.mat')





up = zeros(1, length(t));
vp = zeros(1, length(t));
rp = zeros(1, length(t));


for k=1:length(t)
    if k>1 
        up(k)=(vel_u(k)- vel_u(k-1))/ts;
        vp(k)=(vel_v(k)- vel_v(k-1))/ts;
        rp(k)=(vel_r(k)- vel_r(k-1))/ts;
    else
        up(k)=0;   
        vp(k)=0; 
        rp(k)=0; 
         
    end
end

landa = 100000000000;%lambda
F1=tf(landa,[1 landa]);

u_f=lsim(F1,vel_u,t)';
v_f=lsim(F1,vel_v,t)';
r_f=lsim(F1,vel_r,t)';

up_f=lsim(F1,up,t)';
vp_f=lsim(F1,vp,t)';
rp_f=lsim(F1,rp,t)';


T_u_f=lsim(F1,T_u,t)';
T_r_f=lsim(F1,T_r,t)';


nup = [up; vp; rp];
nu = [vel_u; vel_v; vel_r];
t_ref = [T_u; T_v ; T_r];
T_v=T_u_f*0;
nu_f = [u_f; v_f; r_f];
nup_f = [up_f; vp_f; rp_f];
t_ref_f = [T_u_f; T_v; T_r_f];


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
save('delta_valores_2.mat', 'delta');

% Imprime los valores de los parámetros identificados con un formato personalizado
disp('Parámetros identificados (delta):');
for i = 1:length(delta)
    fprintf('delta_%d es %.4f\n', i, delta(i));
end