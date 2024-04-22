%% create path planning
global Xr Yr  euler_angles first_callback map num %Zr qX qY qZ qW aLX aLY aLZ aAX aAY aAZ
num=1;

% Define execution rate 
fr = 5;
rate = rosrate(fr);

map = [];
% Create publisher to send velocity commands to the robot
cmdPub = rospublisher('/boat/cmd', 'geometry_msgs/Twist');

% Initialize velocity message
cmdMsg = rosmessage(cmdPub);

% Create subscriber with defined callback
posSub = rossubscriber('/wamv/sensors/position/p3d_wamv', 'nav_msgs/Odometry', @odometryCallback);
laserSub = rossubscriber('/laser/scan', @laserCallback);

while ~first_callback
    waitfor(rate);
end
mappos = [Xr, Yr];
paso=10;
tope=50;
total=1+(tope/paso);
first_way=[0:paso:tope;zeros(1,total)];
second_way=[tope:-paso:0; 10*ones(1,total)];
block_way=[first_way,[tope;paso],second_way,[0;paso]];
full_way=block_way;
for i=1:30
    full_way=[full_way,[block_way(1,:);block_way(2,:)+i*20]];
end
le=size(full_way, 2);
full_way=[full_way,[first_way(1,:);first_way(2,:)+100]];
count=1;
map = zeros(50000, 3);
next_point=[full_way(1,count),full_way(2,count)];
Xd = full_way(1,count);
Yd = full_way(2,count);
%real_way=[0;0];

% Controller gains
ku = 1.8; 
kw = 1.5; 
ks = 0.08; % Saturation gain
pause(1);

% Initialize figure for plotting vehicle position
figure;
hold on;
grid on; % Turn on grid lines
xlabel('X');
ylabel('Y');
title('USV Position');

plot_handle = plot(Xr,Yr, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'LineWidth', 2);  % Initialize the plot with a red circle
plot(full_way(1,:),full_way(2,:),'b-*')
partes=le/20;
while count<=le
    i=1;
    while i<partes
     mappos = [Xr, Yr];
     % Plot the desired point
     plot(next_point(1), next_point(2), 'go', 'MarkerSize', 10, 'LineWidth', 2); % Plot the desired point in green
     %real_way=[real_way,mappos'];
     Xe = Xd - Xr;
     Ye = Yd - Yr;
     error = sqrt(Xe^2 + Ye^2);
     a_e = atan2(Ye, Xe);
     angular_difference = normalizeAngle(normalizeAngle(a_e) - normalizeAngle(euler_angles(1)));

     if error<=1
         i=i+1;
         count=count+1;
         next_point=[full_way(1,count),full_way(2,count)];
         Xd = full_way(1,count);
         Yd = full_way(2,count);
     else
         	if (Xd==50 || Xd==0 || abs(angular_difference)>0.2)
                ks=0.05; % Saturation gain
            else
                ks = 0.4;
            end               
            % Define control actions
            if error <= 0.3 
                Uref = 0;
                Wref = 0;          
            else

                Uref = ku * tanh(ks * error) * cos(angular_difference);
                Wref = kw * angular_difference + ku * (tanh(ks * error) / error) * sin(angular_difference) * cos(angular_difference);
            end
            if abs(Wref) > 0.7
                Wref = sign(Wref) * 0.7;
            end
                % Publish velocity message
                cmdMsg.Linear.X = Uref;
                cmdMsg.Angular.Z = Wref;
                send(cmdPub, cmdMsg);
                % Wait for the next execution cycle               
     end
    set(plot_handle, 'XData', Xr, 'YData', Yr);
    xlim([-10 60]); 
    ylim([-10 600]); 
    disp(count)
    disp(i)
    waitfor(rate);
    end
    disp("guardado")
    save('last_simulation.mat','map','count');
    %plot(real_way(1,:),real_way(2,:),'red')
    nombre_archivo = 'path.png';
    saveas(gcf, nombre_archivo);
end

Uref = 0;
Wref = 0; 
cmdMsg.Linear.X = Uref;
cmdMsg.Angular.Z = Wref;
send(cmdPub, cmdMsg);
disp("DONE")
save('last_simulation.mat','map','count');
plot(real_way(1,:),real_way(2,:),'red')
nombre_archivo = 'path.png';
saveas(gcf, nombre_archivo);

 % Callback to update received position data
function odometryCallback(src, msg)
    global Xr Yr  qX qY qZ qW  first_callback euler_angles %Zr aLX aLY aLZ aAX aAY aAZ
    Xr = msg.Pose.Pose.Position.X; % Actual X position
    Yr = msg.Pose.Pose.Position.Y; % Actual Y position
    %Zr = msg.Pose.Pose.Position.Z; % Actual Z position
    qX = msg.Pose.Pose.Orientation.X; % Quaternion X 
    qY = msg.Pose.Pose.Orientation.Y; % Quaternion Y 
    qZ = msg.Pose.Pose.Orientation.Z; % Quaternion Z 
    qW = msg.Pose.Pose.Orientation.W; % Quaternion W
    euler_angles = quat2eul([qW, qX, qY, qZ]); % Output ZYX (Yaw Pitch Roll)---q = [w, x, y, z] order used by MATLAB
%     aLX = msg.Twist.Twist.Linear.X; % Linear velocity X in Inertial frame
%     aLY = msg.Twist.Twist.Linear.Y; % Linear velocity Y in Inertial frame
%     aLZ = msg.Twist.Twist.Linear.Z; % Linear velocity Z in Inertial frame
%     aAX = msg.Twist.Twist.Angular.X; % Angular velocity X in Inertial frame
%     aAY = msg.Twist.Twist.Angular.Y; % Angular velocity Y in Inertial frame
%     aAZ = msg.Twist.Twist.Angular.Z; % Angular velocity Z in Inertial frame
    
    % Update position data
    if ~first_callback
        first_callback = true;
    end
end

function laserCallback(src, msg)
   global  Xr Yr map num
   new=[Xr, Yr, 100-msg.Ranges]; % Por ejemplo, toma los primeros tres elementos de ranges
   map(num,:)=new;
   num=num+1;
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