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

landa = 10;%lambda
F1=tf(landa,[1 landa]);

vel_u=lsim(F1,vel_u,t)';
vel_v=lsim(F1,vel_v,t)';
vel_r=lsim(F1,vel_r,t)';


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

save('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/Sindy_Identification/ident_usv_2.mat')
