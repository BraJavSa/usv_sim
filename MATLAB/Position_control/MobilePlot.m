function Mobile_Graph = MobilePlot(dx, dy, angz)
    global Mobile;
    dz = 0;

    % Matriz de rotación z
    Rz = [cos(angz), -sin(angz), 0, 0; sin(angz), cos(angz), 0, 0; 0, 0, 1, 0; 0, 0, 0, 1];
    Rot_TheWholeArm = Rz;
    Rot_TheWholeArm(1:3, 4) = [dx, dy, dz]';

    BodyColor = 'g';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CUERPO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tam = 0;
    for ii = 1:(length(Mobile.Base))
        base_mat = [Mobile.Base{ii}; ones(1, size(Mobile.Base{ii}, 2))]; % Convertir a coordenadas homogéneas
        robotPatch = Rot_TheWholeArm * base_mat;
        Mobile_Graph(tam + ii) = patch(robotPatch(1,:), robotPatch(2,:), robotPatch(3,:), BodyColor, 'LineWidth', 1.8);
    end

    Rot_TheWholeArm(1:3, 4) = [dx - 0.38*sin(angz), dy + 0.38*cos(angz), dz]';

    tam = tam + ii;
end
