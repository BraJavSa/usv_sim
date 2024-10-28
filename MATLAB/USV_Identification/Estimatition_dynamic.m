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
% %% SIMULATION DYNAMICS
% v_estimate = v;
% for k=1:length(t)
%     v_estimate(:, k+1) = system_dynamic(x, v_estimate(:,k), vref(:,k), ts);
% end
% 
% % figure
% % set(gcf, 'PaperUnits', 'inches');
% % set(gcf, 'PaperSize', [4 2]);
% % set(gcf, 'PaperPositionMode', 'manual');
% % set(gcf, 'PaperPosition', [0 0 10 4]);
% % subplot(4,1,1)
% % plot(t(1:length(ul_ref)),T_u,'Color',[226,76,44]/255,'linewidth',1); hold on
% % plot(t,ul,'--','Color',[226,76,44]/255,'linewidth',1); hold on
% % grid('minor')
% % grid on;
% % legend({'$\mu_{lref}$','$\mu_{l}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
% % legend('boxoff')
% % %title('$\textrm{Identification signals and real Signals}$','Interpreter','latex','FontSize',9);
% % ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
% % 
% % subplot(4,1,2)
% % plot(t(1:length(ul_ref)),um_ref,'Color',[46,188,89]/255,'linewidth',1); hold on
% % plot(t,um,'--','Color',[46,188,89]/255,'linewidth',1); hold on
% % grid('minor')
% % grid on;
% % legend({'$\mu_{mref}$','$\mu_{m}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
% % legend('boxoff')
% % ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
% % 
% % subplot(4,1,3)
% % plot(t(1:length(ul_ref)),un_ref,'Color',[26,115,160]/255,'linewidth',1); hold on
% % plot(t,un,'--','Color',[26,115,160]/255,'linewidth',1); hold on
% % grid('minor')
% % grid on;
% % legend({'$\mu_{nref}$','$\mu_{n}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
% % legend('boxoff')
% % ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
% % 
% % subplot(4,1,4)
% % plot(t(1:length(ul_ref)),w_ref,'Color',[83,57,217]/255,'linewidth',1); hold on
% % plot(t,w,'--','Color',[83,57,217]/255,'linewidth',1); hold on
% % grid('minor')
% % grid on;
% % legend({'$\omega_{ref}$','$\omega$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
% % legend('boxoff')
% % ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
% % xlabel('$\textrm{Time}[s]$','Interpreter','latex','FontSize',9);
% % print -dpng Data_validation
% % print -depsc Data_validation
% % 
% % figure
% % set(gcf, 'PaperUnits', 'inches');
% % set(gcf, 'PaperSize', [4 2]);
% % set(gcf, 'PaperPositionMode', 'manual');
% % set(gcf, 'PaperPosition', [0 0 10 4]);
% % subplot(4,1,1)
% % plot(t,v_estimate(1,1:length(t)),'Color',[226,76,44]/255,'linewidth',1); hold on
% % plot(t,ul,'--','Color',[226,76,44]/255,'linewidth',1); hold on
% % grid('minor')
% % grid on;
% % legend({'$\mu_{lm}$','$\mu_{l}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
% % legend('boxoff')
% % %title('$\textrm{Dynamic Model Identification}$','Interpreter','latex','FontSize',9);
% % ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
% % 
% % subplot(4,1,2)
% % plot(t,v_estimate(2,1:length(t)),'Color',[46,188,89]/255,'linewidth',1); hold on
% % plot(t,um,'--','Color',[46,188,89]/255,'linewidth',1); hold on
% % grid('minor')
% % grid on;
% % legend({'$\mu_{mm}$','$\mu_{m}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
% % legend('boxoff')
% % ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
% % 
% % subplot(4,1,3)
% % plot(t,v_estimate(3,1:length(t)),'Color',[26,115,160]/255,'linewidth',1); hold on
% % plot(t,un,'--','Color',[26,115,160]/255,'linewidth',1); hold on
% % grid('minor')
% % grid on;
% % legend({'$\mu_{nm}$','$\mu_{n}$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
% % legend('boxoff')
% % ylabel('$[m/s]$','Interpreter','latex','FontSize',9);
% % 
% % subplot(4,1,4)
% % plot(t,v_estimate(4,1:length(t)),'Color',[83,57,217]/255,'linewidth',1); hold on
% % plot(t,w,'--','Color',[83,57,217]/255,'linewidth',1); hold on
% % grid('minor')
% % grid on;
% % legend({'$\omega_{m}$','$\omega$'},'Interpreter','latex','FontSize',11,'Orientation','horizontal');
% % legend('boxoff')
% % ylabel('$[rad/s]$','Interpreter','latex','FontSize',9);
% % xlabel('$\textrm{Time}[s]$','Interpreter','latex','FontSize',9);
% print -dpng Data_model
% print -depsc Data_model
% save('parameters.mat','chi')