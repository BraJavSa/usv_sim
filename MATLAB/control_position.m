% Clear all existing variables
clc
clear
close all

global Xr Yr Zr qX qY qZ qW aLX aLY aLZ aAX aAY aAZ euler_angles first_callback

% Desired position
Xd = -20; 
Yd = 20;
% Define execution rate 
fr = 4;
rate = rosrate(fr);
ts = 1/fr;
tf = 40;
t = 0:ts:tf;
tr = 0;

% Create publisher to send velocity commands to the robot
cmdPub = rospublisher('/boat/cmd', 'geometry_msgs/Twist');

% Initialize velocity message
cmdMsg = rosmessage(cmdPub);

% Create subscriber with defined callback
posSub = rossubscriber('/wamv/sensors/position/p3d_wamv', 'nav_msgs/Odometry', @odometryCallback);

while ~first_callback
    waitfor(rate);
end

% Controller gains
ku = 1.8; 
kw = 1.5; 
ks = 0.08; % Saturation gain

% Initialize figure for plotting vehicle position
figure;
hold on;
grid on; % Turn on grid lines
xlabel('X');
ylabel('Y');
title('USV Position');

% Plot the desired point
plot(Xd, Yd, 'go', 'MarkerSize', 10, 'LineWidth', 2); % Plot the desired point in green

disp("Position Controller")
xrk(1) = Xr;
yrk(1) = Yr;
% Calculate position error
Xe(1) = Xd - Xr;
Ye(1) = Yd - Yr;
error(1) = sqrt(Xe(1)^2 + Ye(1)^2);
plot_handle = plot(xrk(1),yrk(1), 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'LineWidth', 2);  % Initialize the plot with a red circle
Uref(1)=0;
Wref(1)=0;
% Main loop
for k = 2:length(t)
    xrk(k) = Xr;
    yrk(k) = Yr;
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
    

    if abs(Wref(k)) > 0.7
        Wref(k) = sign(Wref(k)) * 0.7;
    end
    
    
    set(plot_handle, 'XData', xrk(k), 'YData', yrk(k));
    xlim([-25 25]); 
    ylim([-25 25]);  

    % Publish velocity message
    cmdMsg.Linear.X = Uref(k);
    cmdMsg.Angular.Z = Wref(k);
    send(cmdPub, cmdMsg);
    % Wait for the next execution cycle
    waitfor(rate);
end
Uref(length(t)) = 0;
Wref(length(t)) = 0; 
cmdMsg.Linear.X = Uref(length(t));
cmdMsg.Angular.Z = Wref(length(t));
send(cmdPub, cmdMsg);
disp("DONE")

% Plot error evolution over time
figure;
plot(t, error, 'b', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Error');
title('Error Evolution');

% Plot error evolution over time
figure;
subplot(2, 1, 1);
plot(t, Uref, 'b', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Uref');
title('Uref Evolution');

subplot(2, 1, 2);
plot(t, Wref, 'r', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Wref');
title('Wref Evolution');

% Callback to update received position data
function odometryCallback(src, msg)
    global Xr Yr Zr qX qY qZ qW aLX aLY aLZ aAX aAY aAZ first_callback euler_angles
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
