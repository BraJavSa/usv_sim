% Clear all existing variables
clc
clear
close all
% Define execution rate (10 Hz)
fr = 5;
rate = rosrate(fr);

ts = 1/fr;
tf = 15;
t = 0:ts:tf;
tr=0;
% Create publisher to send velocity commands to the robot
cmdPub = rospublisher('/boat/cmd', 'geometry_msgs/Twist');

% Initialize velocity message
cmdMsg = rosmessage(cmdPub);




% Initialize global variables for position, orientation, and velocity
global Xr Yr Zr qX qY qZ qW aLX aLY aLZ aAX aAY aAZ euler_angles Uref Wref first_callback
Xr = 0; Yr = 0; Zr = 0; qX = 0; qY = 0; qZ = 0; qW = 0; aLX = 0; aLY = 0; aLZ = 0; aAX = 0; aAY = 0; aAZ = 0; euler_angles = [0,0,0]; primer_callback = false; Uref = 0; Wref = 0;


% Create subscriber with defined callback
posSub = rossubscriber('/wamv/sensors/position/p3d_wamv', 'nav_msgs/Odometry', @odometryCallback);

while ~first_callback
    waitfor(rate);
end



% Desired position
Xd = 25; 
Yd = 25;

% Controller gains
ku = 1.8; 
kw = 1.5; 
ks = 0.2; % Saturation gain

% Initialize figure for plotting vehicle position
 fig=figure('Name','Simulacion');
 set(fig,'position',[60 60 980 600]);
 axis square; cameratoolbar
 axis([-1 50 -1 50 0 1]);
 grid on
 MobileRobot;
 M1=MobilePlot(Xr,Yr,normalizeAngle(euler_angles(1)));hold on
 M2=plot(Xr,Yr,'b','LineWidth',2);
 
disp("Position Controller")
% Main loop
for k = 1:length(t)
    xrk(k)=Xr;
    yrk(k)=Yr;
    % Calculate position error
    Xe(k) = Xd - Xr;
    Ye(k) = Yd - Yr;
    error(k) = sqrt(Xe(k)^2 + Ye(k)^2);
    a_e(k) = atan2(Ye(k), Xe(k));
    angular_difference(k) = normalizeAngle(normalizeAngle(a_e(k)) - normalizeAngle(euler_angles(1)));
    
    % Define control actions
    if error(k) <= 0.5 
        Uref(k) = 0;
        Wref(k) = 0;          
    else
        Uref(k) = ku * tanh(ks * error(k)) * cos(angular_difference(k));
        Wref(k) = kw * angular_difference(k) + ku * (tanh(ks * error(k)) / error(k)) * sin(angular_difference(k)) * cos(angular_difference(k));
    end
     delete (M1)
     delete (M2)
     M1=MobilePlot(xrk(k),yrk(k),euler_angles(1)); hold on
     M2=plot(xrk(2:k),yrk(2:k),'b','LineWidth',2);   
     waitfor(rate);
    % Publish velocity message
    cmdMsg.Linear.X = Uref(k);
    cmdMsg.Angular.Z = Wref(k);
    send(cmdPub, cmdMsg);
    tr=tr+ts;
    disp(t)
    % Wait for the next execution cycle
    waitfor(rate);
end
Uref(k) = 0;
Wref(k) = 0; 
cmdMsg.Linear.X = Uref(k);
cmdMsg.Angular.Z = Wref(k);
send(cmdPub, cmdMsg);
disp("DONE")

% Callback to update received position data
function odometryCallback(src, msg)
    global Xr Yr Zr qX qY qZ qW aLX aLY aLZ aAX aAY aAZ euler_angles first_callback

    Xr = msg.Pose.Pose.Position.X; % Actual X position
    Yr = msg.Pose.Pose.Position.Y; % Actual Y position
    Zr = msg.Pose.Pose.Position.Z; % Actual Z position
    qX = msg.Pose.Pose.Orientation.X; % Quaternion X 
    qY = msg.Pose.Pose.Orientation.Y; % Quaternion Y 
    qZ = msg.Pose.Pose.Orientation.Z; % Quaternion Z 
    qW = msg.Pose.Pose.Orientation.W; % Quaternion W
    q = [qW, qX, qY, qZ];
    euler_angles = quat2eul(q); % Output ZYX (Yaw Pitch Roll)---q = [w, x, y, z] order used by MATLAB
    
    aLX = msg.Twist.Twist.Linear.X; % Linear velocity X in Inertial frame
    aLY = msg.Twist.Twist.Linear.Y; % Linear velocity Y in Inertial frame
    aLZ = msg.Twist.Twist.Linear.Z; % Linear velocity Z in Inertial frame
    aAX = msg.Twist.Twist.Angular.X; % Angular velocity X in Inertial frame
    aAY = msg.Twist.Twist.Angular.Y; % Angular velocity Y in Inertial frame
    aAZ = msg.Twist.Twist.Angular.Z; % Angular velocity Z in Inertial frame
    
    % Update position data
    
    if ~first_callback
        first_callback = true;
    end
end

function n_angle = normalizeAngle(angle)
  if angle > pi
        angle = angle - 2*pi;
    elseif angle < -pi
        angle = angle + 2*pi;
    else
        % No action needed
  end
    n_angle = angle;
end
