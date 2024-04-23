load('last_simulation.mat')

% Convert the matrices to double type
x = double(map(:, 1));
y = double(map(:, 2));
depth = double(map(:, 3));

% Create a grid of coordinates for the bathymetric map
[X, Y] = meshgrid(linspace(min(x), max(x), 100), linspace(min(y), max(y), 100));

% Interpolate the depth on the coordinate grid
Z = griddata(x, y, depth, X, Y);

% Create the 3D bathymetric map
figure('units','normalized','outerposition',[0 0 1 1]); % Maximize the figure
surf(X, Y, Z-100);
colorbar;
xlabel('X Position');
ylabel('Y Position');
zlabel('Depth');
title('3D Bathymetric Map');

% Configure the VideoWriter object
writerObj = VideoWriter('bathymetric_map.avi','Uncompressed AVI'); % Uncompressed video format for better quality
writerObj.FrameRate = 10; % Video frame rate
open(writerObj);

% Record each frame of the bathymetric map with constant rotation in Z
for i = 1:150 % Number of frames to record
    % Modify the view to rotate the bathymetric map in Z
    view(30+i*4, 30); % Constant rotation in Z
    % Capture the current frame and add it to the video
    writeVideo(writerObj, getframe(gcf));
end

% Close the VideoWriter object
close(writerObj);

% Crear el mapa batimétrico
figure;
contourf(X, Y, Z);
colorbar;
xlabel('Posición X');
ylabel('Posición Y');
title('2D Bathymetric Map');
name = 'path.png';
saveas(gcf, name);
