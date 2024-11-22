%% clear variables
clc, clear all, close all;
clear tf;




%% LOAD VALUES FROM MATRICES
load('ident_usv.mat')
acc_u = zeros(1, length(t));
acc_v = zeros(1, length(t));
acc_r = zeros(1, length(t));


for k=1:length(t)
    if k>1 
        acc_u(k)=(vel_u(k)- vel_u(k-1))/ts;
        acc_v(k)=(vel_v(k)- vel_v(k-1))/ts;
        acc_r(k)=(vel_r(k)- vel_r(k-1))/ts;
    else
        acc_u(k)=0;   
        acc_v(k)=0; 
        acc_r(k)=0; 
         
    end
end

pass_vel_u=circshift(vel_u, 1);
pass_vel_u(1) = 0;
pass_vel_v=circshift(vel_v, 1);
pass_vel_v(1) = 0;
pass_vel_r=circshift(vel_r, 1);
pass_vel_r(1) = 0;

pass2_vel_u=circshift(vel_u, 2);
pass2_vel_u(1) = 0;
pass2_vel_u(2) = 0;
pass2_vel_v=circshift(vel_v, 2);
pass2_vel_v(1) = 0;
pass2_vel_v(2) = 0;
pass2_vel_r=circshift(vel_r, 2);
pass2_vel_r(1) = 0;
pass2_vel_r(2) = 0;

pass_T_u=circshift(T_u, 1);
pass_T_u(1) = 0;
pass_T_r=circshift(T_r, 1);
pass_T_r(1) = 0;

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


save('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/ident_usv_2.mat')
