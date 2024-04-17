% Clear all existing variables
clc
clear
close all

% Define execution rate (10 Hz)
fr = 10;
rate = rosrate(fr);

ts = 1/fr;
tf = 20;
t = 0:ts:tf;

% Create publisher to send velocity commands to the robot
cmdPub = rospublisher('/boat/cmd', 'geometry_msgs/Twist');

% Initialize velocity message
cmdMsg = rosmessage(cmdPub);


% Initialize global variables for position, orientation, and velocity
global Xr Yr Zr qX qY qZ qW aLX aLY aLZ aAX aAY aAZ euler_angles Uref Wref first_callback
Xr = 0; Yr = 0; Zr = 0; qX = 0; qY = 0; qZ = 0; qW = 0; aLX = 0; aLY = 0; aLZ = 0; aAX = 0; aAY = 0; aAZ = 0; euler_angles = [0,0,0]; first_callback = false; Uref = 0; Wref = 0;

% Create subscriber with defined callback
posSub = rossubscriber('/wamv/sensors/position/p3d_wamv', 'nav_msgs/Odometry', @odometryCallback);

while ~first_callback
    waitfor(rate);
end

% Desired position
Xd = 0; 
Yd = 0;

% Controller gains
ku = 1.8; 
kw = 1.5; 
ks = 0.2; % Saturation gain

%xrd = 40*sin(0.0004*t);                           
%yrd = 40*sin(0.0002*t);                               
%xrdp= 40*cos(0.0004*t)*0.4;  
%yrdp= 40*cos(0.0002*t)*0.2;

xrd = 40 * ones(size(t));                           
yrd = 40 * ones(size(t));                                
xrdp= 0 * ones(size(t));   
yrdp= 0 * ones(size(t)); 

kx=0.8;
ky=0.5;
a=0.2;


disp("Trackin Controller")

% Main loop
for k = 1:length(t)
    % Calculate position error

    Xe(k) = Xd - Xr;
    Ye(k) = Yd - Yr;    
    vx=xrdp(k)+kx*(Xe(k));
    vy=yrdp(k)+ky*(Ye(k));
    vpsi=-(vx/a)*sin(euler_angles(1))+(vy/a)*cos(normalizeAngle(euler_angles(1)));
    Uref(k)=vx*cos(euler_angles(1))+vy*sin(euler_angles(1));
    Wref(k)=vpsi;        
    error(k) = sqrt(Xe(k)^2 + Ye(k)^2);
    
    
    % Publish velocity message
    cmdMsg.Linear.X = Uref(k);
    cmdMsg.Angular.Z = Wref(k);
    send(cmdPub, cmdMsg);
    vel=[Uref(k),Wref(k)];
    disp(vel);
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
    
    aLX = msg.Twist.Twist.Linear.X; % Linear velocity X in body frame
    aLY = msg.Twist.Twist.Linear.Y; % Linear velocity Y in body frame
    aLZ = msg.Twist.Twist.Linear.Z; % Linear velocity Z in body frame
    aAX = msg.Twist.Twist.Angular.X; % Angular velocity X in body frame
    aAY = msg.Twist.Twist.Angular.Y; % Angular velocity Y in body frame
    aAZ = msg.Twist.Twist.Angular.Z; % Angular velocity Z in body frame
    
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
