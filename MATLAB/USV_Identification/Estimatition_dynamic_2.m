%% ESTIMATION FO PARAMETERS DORNE DYNAMIC %%

%% clear variables
clc, clear all, close all;

%% LOAD VALUES FROM MATRICES

%load('ident_usv.mat', 'T_u', 'T_r','T_v', 'vel_u', 'vel_v', 'vel_r','ts');

%% REFERENCE SIGNALS
load('usv_data2.mat')



ts=1/50;
vel_u=odom_lin_vel_x;
vel_v=odom_lin_vel_y;
vel_r=odom_ang_vel_z;
duracion=ts*length(vel_u);
t = 0:ts:duracion-ts; % Vector de tiempo

d=1.4;
% Aplica la función calcularFuerzaPropulsor para obtener los torques en cada paso
for i = 1:floor(duracion / ts)

    % Calcula las fuerzas de cada propulsor
    T_right_thruster = calcularFuerzaPropulsor(calcularFuerzaPropulsor(right_thruster(i))) * 2;
    T_left_thruster = calcularFuerzaPropulsor(calcularFuerzaPropulsor(left_thruster(i))) * 2;

    % Calcula las fuerzas de traslación y rotación
    T_u(i) = T_left_thruster + T_right_thruster;  % Fuerza de traslación
    T_r(i) = d * (-T_left_thruster + T_right_thruster);  % Fuerza de rotación
     % No hay fuerza lateral en este caso
end
T_u=[0, T_u(1:end-1)];
T_r=[0, T_r(1:end-1)];
T_v=T_u*0;
tau_ref = [T_u;T_v; T_r];
%% SYSTEM TIME

t=0:ts:(length(T_u)-1)*ts;
N = length(t);
%% SYSTEM SIGNALS

vel_u = vel_u(1,1:length(T_u));
vel_v = vel_v(1,1:length(T_v));
vel_r = vel_r(1,1:length(T_r));

%% ACELERATION SYSTEM
up = [0 , diff(vel_u)/ts];
vp = [0 , diff(vel_v)/ts];
rp = [0 , diff(vel_r)/ts];


vel_p = [up; vp; rp];

landa = 1000000;%lambda
F1=tf(landa,[1 landa]);


vel_u=lsim(F1,vel_u,t)';
vel_v=lsim(F1,vel_v,t)';
vel_r=lsim(F1,vel_r,t)';

up=lsim(F1,up,t)';
vp=lsim(F1,vp,t)';
rp=lsim(F1,rp,t)';


T_u=lsim(F1,T_u,t)';
T_v=lsim(F1,T_v,t)';
T_r=lsim(F1,T_r,t)';

vel = [vel_u; vel_v; vel_r];
vel_p = [up; vp; rp];
tau_ref = [T_u;T_v; T_r];


%% Parametros del optimizador
options = optimset('Display','iter',...
                'TolFun', 1e-8,...
                'MaxIter', 10000,...
                'Algorithm', 'active-set',...
                'FinDiffType', 'forward',...
                'RelLineSrchBnd', [],...
                'RelLineSrchBndDuration', 1,...
                'TolConSQP', 1e-6); 
x0=zeros(1,11)+0.1;           
f_obj1 = @(x)  cost_func_dynamic(x, tau_ref, vel_p, vel, N);
x = fmincon(f_obj1,x0,[],[],[],[],[],[],[],options);
chi = x;