%% clear variables
clc, clear all, close all;
clear tf;




%% LOAD VALUES FROM MATRICES
load('ident_usv_3.mat')
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

landa = 10;%lambda
F1=tf(landa,[1 landa]);

vel_u=lsim(F1,vel_u,t)';
vel_v=lsim(F1,vel_v,t)';
vel_r=lsim(F1,vel_r,t)';

acc_u=lsim(F1,up,t)';
acc_v=lsim(F1,vp,t)';
acc_r=lsim(F1,rp,t)';

T_u=lsim(F1,T_u,t)';
T_r=lsim(F1,T_r,t)';
T_v=lsim(F1,T_v,t)';

pass_vel_u=circshift(vel_u, 1);
pass_vel_u(1) = 0;
pass_vel_v=circshift(vel_v, 1);
pass_vel_v(1) = 0;
pass_vel_r=circshift(vel_r, 1);
pass_vel_r(1) = 0;


pass2_vel_u=circshift(pass_vel_u, 1);
pass2_vel_u(1) = 0;
pass2_vel_v=circshift(pass_vel_v, 1);
pass2_vel_v(1) = 0;
pass2_vel_r=circshift(pass_vel_r, 1);
pass2_vel_r(1) = 0;




% Suponiendo que ya tienes los vectores vel_u, vel_v, vel_r, acc_u, acc_v, acc_r, T_u, T_r y T_v

% Suponiendo que ya tienes los vectores vel_u, vel_v, vel_r, acc_u, acc_v, acc_r, T_u, T_r y T_v

% Desplazar valores a la posición k-2 utilizando circshift
vel_u = circshift(vel_u, 1);
vel_v = circshift(vel_v, 1);
vel_r = circshift(vel_r, 1);
acc_u = circshift(acc_u, 1);
acc_v = circshift(acc_v, 1);
acc_r = circshift(acc_r, 1);

% Eliminar los últimos 10 valores
vel_u = vel_u(4:end-10);
vel_v = vel_v(4:end-10);
vel_r = vel_r(4:end-10);
acc_u = acc_u(4:end-10);
acc_v = acc_v(4:end-10);
acc_r = acc_r(4:end-10);
T_u = T_u(4:end-10);
T_r = T_r(4:end-10);
T_v = T_v(4:end-10);


save('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/ident_usv_3.mat')
