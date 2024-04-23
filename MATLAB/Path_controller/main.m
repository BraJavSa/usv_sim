% Clear all existing variables
clc
clear
close all;

global Xr Yr euler_angles  %Zr qX qY qZ qW aLX aLY aLZ aAX aAY aAZ

% Desired position
 xrd = 40*sin(0.004*t);                           %Posición x
 yrd = 40*sin(0.002*t);                               %Posición y 
 xrdp= 40*cos(0.004*t)*0.4;  
 yrdp= 40*cos(0.002*t)*0.2;

% Define execution rate 
fr = 4;
rate = rosrate(fr);
ts = 1/fr;
tf = 65;
t = 0:ts:tf;

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
Xe(1) = Xd - Xr;
Ye(1) = Yd - Yr;
error(1) = sqrt(Xe(1)^2 + Ye(1)^2);
[errorPlot UrefPlot WrefPlot]=init_plot(Uref(1),Wref(1),error(1),tf);
M1=init_robot_plot(xrk(1),yrk(1),psir(1),Xd,Yd);

disp("Position Controller")
for k = 2:length(t)
        
        %Controller
        [Uref(k), Wref(k),error(k),xrk(k), yrk(k),psir(k)] = controller(Xd, Yd, Xr, Yr, euler_angles);
        
        %Figure plotting
        M1=update_robot_plot(M1,xrk(k),yrk(k),psir(k));  
        [errorPlot UrefPlot WrefPlot]=update_plot(errorPlot,UrefPlot,WrefPlot,error,Uref,Wref,t,k);
       

        % publish control actions
        cmdMsg.Linear.X = Uref(k);
        cmdMsg.Angular.Z = Wref(k);
        send(cmdPub, cmdMsg);

        % Wait for the next execution cycle
        waitfor(rate);
 end
disp("DONE")



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
