load('last_simulation (copy).mat')

% Convert the matrices to double type
x = double(map(:, 1));
y = double(map(:, 2));
depth = double(map(:, 3));

% Create a grid of coordinates for the bathymetric map
[X, Y] = meshgrid(linspace(min(x), max(x), 100), linspace(min(y), max(y), 100));

% Initialize an empty array to store the interpolated depths
Z = zeros(size(X));

% Create the 3D bathymetric map figure
figure('units','normalized','outerposition',[0 0 1 1]); % Maximize the figure

% Set the limits for x, y, and z axes
xlim([-5, 55]);
ylim([-5, 300]);
zlim([-80, 1]);

surf(X, Y, Z-100);
colorbar;
xlabel('X Position');
ylabel('Y Position');
zlabel('Depth');
title('3D Bathymetric Map');

hold on; % Hold the plot to add the boat position and sonar cone

% Initialize the boat position plot
boat_plot = scatter3(0, 0, 0, 'ro', 'filled');

% Initialize the sonar cone
sonar_resolution = 20; % Resolution of the sonar cone
sonar_color = [0, 1, 0]; % Color of the sonar cone
sonar_alpha = 0.5; % Transparency of the sonar cone
sonar_patch = patch('Faces', [], 'Vertices', [], ...
    'FaceColor', sonar_color, 'FaceAlpha', sonar_alpha, 'EdgeColor', 'green'); % Set EdgeColor to 'none'

% Plot the translucent blue water surface
water_surface = fill3([min(x), min(x), max(x), max(x)], [min(y), max(y), max(y), min(y)], [0, 0, 0, 0], 'b', 'FaceAlpha', 0.1);

% Interpolate the depth on the coordinate grid one point at a time
for i = 1:numel(x)
    % Interpolate depth for the current point
    Z_new = griddata(x(1:i), y(1:i), depth(1:i), X, Y);
    
    % Update the Z values of the plot
    set(findobj(gca, 'type', 'surface'), 'ZData', Z_new-100);    
    xlim([-10, 60]);
    ylim([-10, 400]);
    zlim([-80, 1]);
    
    % Update boat position
    set(boat_plot, 'XData', x(i), 'YData', y(i), 'ZData', sin(i*0.02));
    
    % Determine sonar length based on depth at current point
    sonar_length = depth(i)-110;
    [X_sonar, Y_sonar, Z_sonar] = cylinder([0, 1], sonar_resolution);
    Z_sonar = Z_sonar * sonar_length;
    X_sonar = X_sonar + x(i);
    Y_sonar = Y_sonar + y(i);
    
    % Update sonar cone position and length
    set(sonar_patch, 'Vertices', [X_sonar(:), Y_sonar(:), Z_sonar(:)], 'Faces', 1:size(X_sonar, 2));
    
    % Update water surface
    set(water_surface, 'XData', [min(x), min(x), max(x), max(x)], 'YData', [min(y), max(y), max(y), min(y)], 'ZData', [sin(i*0.02), sin(i*0.02), sin(i*0.02), sin(i*0.02)]);

    view(30+i/5, 30);
    % Pause for 1 second
    pause(0.001);
end

hold off; % Release the plot hold
disp("DONE")
