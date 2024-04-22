% Clear all existing variables
clc
clear
close all;

global Xr Yr euler_angles first_callback 

% Desired position
Xd = 15; 
Yd = -17;

% Define execution rate 
fr = 4;
rate = rosrate(fr);
ts = 1/fr;
tf = 48;
t = 0:ts:tf;

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

disp("Position Controller")
xrk(1) = Xr;
yrk(1) = Yr;
psir(1) = euler_angles(1);

% Calculate position error
Xe(1) = Xd - Xr;
Ye(1) = Yd - Yr;
error(1) = sqrt(Xe(1)^2 + Ye(1)^2);
Uref(1) = 0;
Wref(1) = 0;

% Plot the desired position
fig1 = figure('Name','Robot Movement');
set(fig1,'position',[60 60 980 600]);
axis square; cameratoolbar
axis([-20 20 -20 20 0 1]);
grid on
MobileRobot;
M1 = MobilePlot(xrk(1), yrk(1), psir(1));
hold on
scatter(Xd, Yd, 'r', 'filled'); % punto rojo en la posición deseada

% Plot the error and control actions
fig2 = figure('Name', 'Error and Control Actions');
set(fig2,'position',[60 60 980 600]);
subplot(3, 1, 1);
errorPlot = plot(0, 0, 'b', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Error');
grid on;
title('Error Evolution');
axis([0 tf 0 50]); % Limitar los ejes en el subplot

subplot(3, 1, 2);
UrefPlot = plot(0, 0, 'b', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Uref');
grid on;
title('Uref Evolution');
axis([0 tf -2 2]); % Limitar los ejes en el subplot

subplot(3, 1, 3);
WrefPlot = plot(0, 0, 'r', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Wref');
grid on;
title('Wref Evolution');
axis([0 tf -1 1]); % Limitar los ejes en el subplot

pause(8)
% Main loop
for k = 2:length(t)
    xrk(k) = Xr;
    yrk(k) = Yr;
    psir(k) = euler_angles(1);
    
    % Calculate position error
    Xe(k) = Xd - Xr;
    Ye(k) = Yd - Yr;
    error(k) = sqrt(Xe(k)^2 + Ye(k)^2);
    a_e(k) = atan2(Ye(k), Xe(k));
    angular_difference(k) = normalizeAngle(a_e(k) - euler_angles(1));
    
    % Define control actions
    if error(k) <= 0.2 
        Uref(k) = 0;
        Wref(k) = 0;          
    else
        Uref(k) = ku * tanh(ks * error(k)) * cos(angular_difference(k));
        Wref(k) = kw * angular_difference(k) + ku * (tanh(ks * error(k)) / error(k)) * sin(angular_difference(k)) * cos(angular_difference(k));
    end
    
    if abs(Wref(k)) > 0.7
        Wref(k) = sign(Wref(k)) * 0.7;
    end
    
    % Update robot movement plot
    delete(M1)
    M1 = MobilePlot(xrk(k), yrk(k), psir(k));
    hold on

    % Update error and control actions plot
    set(errorPlot, 'XData', t(1:k), 'YData', error(1:k));
    set(UrefPlot, 'XData', t(1:k), 'YData', Uref(1:k));
    set(WrefPlot, 'XData', t(1:k), 'YData', Wref(1:k));

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

% Callback to update received position data
function odometryCallback(src, msg)
    global Xr Yr first_callback euler_angles
    Xr = msg.Pose.Pose.Position.X; % Actual X position
    Yr = msg.Pose.Pose.Position.Y; % Actual Y position
    euler_angles = quat2eul([msg.Pose.Pose.Orientation.W, msg.Pose.Pose.Orientation.X, ...
                             msg.Pose.Pose.Orientation.Y, msg.Pose.Pose.Orientation.Z]); % Output ZYX (Yaw Pitch Roll)
    
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
    end
    n_angle = angle;
end
