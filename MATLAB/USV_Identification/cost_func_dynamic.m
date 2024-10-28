function [cost] = cost_func_dynamic(x, tau_ref, vel_p, vel, N)
%UNTITLED6 Summary of this function goes here
tau_ref_system = open_loop_dynamic(x, vel_p, vel, N);
he = error_dynamic(tau_ref, tau_ref_system, N);
cost = he'*he;
end

