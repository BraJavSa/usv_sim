function [tau] = open_loop_dynamic(x, vel_p, vel, N)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
for k=1:N
   tau(:,k) = dynamic_identification(x, vel_p(:,k), vel(:,k)); 
end
end

