function [he] = error_dynamic(tau_ref, vref_system, N)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
he = [];
for k=1:N
    he = [he; tau_ref(:,k)-vref_system(:,k)];
end
end

