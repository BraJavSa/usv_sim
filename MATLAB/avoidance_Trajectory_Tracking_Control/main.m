% Clear all existing variables
clc
clear
close all;

global Xr Yr euler_angles  %Zr qX qY qZ qW aLX aLY aLZ aAX aAY aAZ

 
% Define execution rate 
fr = 10;
rate = rosrate(fr);
ts = 1/fr;
tf = 120;
t = 0:ts:tf;

% Desired position
 Xd = 40*sin(0.04*t);                           
 Yd = 40*sin(0.02*t);                                
 Xdp= 40*cos(0.04*t)*0.04;  
 Ydp= 40*cos(0.02*t)*0.02;
 

% Create publisher to send velocity commands to the robot
cmdPub = rospublisher('/boat/cmd', 'geometry_msgs/Twist');

% Initialize velocity message
cmdMsg = rosmessage(cmdPub);

% Create subscriber with defined callback
posSub = rossubscriber('/wamv/sensors/position/p3d_wamv', 'nav_msgs/Odometry', @odometryCallback);

for i = 1:3
    pause(0.1);
end

% Initial conditions
Uref(1) = 0;
Wref(1) = 0;
psir(1) = euler_angles(1);
xrk(1)=Xr;
yrk(1)=Yr;
w_obs(1)=0;
w_obs(2)=0;
 obx=0;
 oby=39.9;
Xe(1) = Xd(1) - xrk(1);
Ye(1) = Yd(1) - xrk(1);
error(1) = sqrt(Xe(1)^2 + Ye(1)^2);
M1=init_robot_plot(xrk(1),yrk(1),psir(1),Xd,Yd);
M2=plot(Xd(1),Yd(1),'bo', 'MarkerSize', 4);
disp("Trajectory Trackin Control")

for k = 3:length(t)
        
        %Controller
        [Uref(k), Wref(k),error(k),xrk(k), yrk(k),psir(k),w_obs(k)] = controller(Xd(k), Yd(k), Xdp(k), Ydp(k),Xr, Yr, euler_angles,w_obs(k-1),w_obs(k-2),ts,obx,oby);
        
        %Figure plotting
        [M1, M2]=update_robot_plot(M1,M2,xrk(k),yrk(k),psir(k),Xd(k), Yd(k),obx,oby);  
        
        % publish control actions
        cmdMsg.Linear.X = Uref(k);
        cmdMsg.Angular.Z = Wref(k);
        send(cmdPub, cmdMsg);

        % Wait for the next execution cycle
        waitfor(rate);
end
 

disp("DONE")
cmdMsg.Linear.X = 0;
cmdMsg.Angular.Z = 0;
send(cmdPub, cmdMsg);

fig3 = figure('Name','Robot Movement');
set(fig3,'position',[60 60 980 600]);
axis square; cameratoolbar
axis([-50 50 -50 50 0 1]);
grid on
plot(Xd,Yd,'r','LineWidth',2);
hold on
plot(xrk,yrk,'b','LineWidth',2)
hold on

figure('Name','Error')
subplot(311)
plot(t,error,'linewidth',2), grid on
legend('Error Evolution')
xlabel('Time [s]'), ylabel('Error  [m]')
title("")
axis([0 tf 0 max(error)+5]); 
subplot(312)
title("CONTROL ACTIONS")

plot(t,Uref,'linewidth',2), grid on
legend('Uref Evolution')
xlabel('Time [s]'), ylabel('Uref [m/s]')

subplot(313)
plot(t,Wref,'g','linewidth',2), grid on
legend('Wref Evolution')
xlabel('Time [s]'), ylabel('Wref [rad/s]')

% Callback to update received position data
function odometryCallback(src, msg)
    global Xr Yr  euler_angles %Zr aLX aLY aLZ aAX aAY aAZ
    Xr = msg.Pose.Pose.Position.X; % Actual X position
    Yr = msg.Pose.Pose.Position.Y; % Actual Y position
    euler_angles = quat2eul([msg.Pose.Pose.Orientation.W, msg.Pose.Pose.Orientation.X, ...
                             msg.Pose.Pose.Orientation.Y, msg.Pose.Pose.Orientation.Z]); % Output ZYX (Yaw Pitch Roll)
%     Zr = msg.Pose.Pose.Position.Z; % Actual Z position
%     qX = msg.Pose.Pose.Orientation.X; % Quaternion X 
%     qY = msg.Pose.Pose.Orientation.Y; % Quaternion Y 
%     qZ = msg.Pose.Pose.Orientation.Z; % Quaternion Z 
%     qW = msg.Pose.Pose.Orientation.W; % Quaternion W
%     aLX = msg.Twist.Twist.Linear.X; % Linear velocity X in Inertial frame
%     aLY = msg.Twist.Twist.Linear.Y; % Linear velocity Y in Inertial frame
%     aLZ = msg.Twist.Twist.Linear.Z; % Linear velocity Z in Inertial frame
%     aAX = msg.Twist.Twist.Angular.X; % Angular velocity X in Inertial frame
%     aAY = msg.Twist.Twist.Angular.Y; % Angular velocity Y in Inertial frame
%     aAZ = msg.Twist.Twist.Angular.Z; % Angular velocity Z in Inertial frame
end
