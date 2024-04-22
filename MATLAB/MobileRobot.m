function TableDrill
    length = 1.2;
    width = 0.8;
    thickness = 0.1;

    global Mobile;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Cuerpo %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Definimos las coordenadas de los vértices del triángulo
    base = [
        length/2, 0, 0;   % Vértice inferior (invertido)
        -length/2, width/2, 0;   % Vértice derecho (invertido)
        -length/2, -width/2, 0    % Vértice izquierdo (invertido)
    ];

    Mobile.Base = cell(1,1);
    Mobile.Base{1} = base';

end 
