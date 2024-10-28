% Cargar datos de entrada y salida
load('ident_usv.mat'); % Asegúrate de que 'ident_usv.mat' contenga T_u, T_v, T_r, vel_u, vel_v, vel_r

% Parámetros de la simulación
Fs = 50; % Frecuencia de muestreo (Hz)
dt = 1 / Fs; % Intervalo de muestreo (s)

% Datos de entrada (torques) y salida (velocidades)
inputData = [T_u, T_v, T_r];
outputData = [vel_u, vel_v, vel_r];

% Derivar las velocidades (aceleraciones)
vel_u_dot = gradient(vel_u, dt);
vel_v_dot = gradient(vel_v, dt);
vel_r_dot = gradient(vel_r, dt);
outputDot = [vel_u_dot, vel_v_dot, vel_r_dot];

% Configuración de la biblioteca de funciones para SINDy
polyorder = 2; % Orden del polinomio para funciones polinómicas (se incluirán términos hasta el orden 2)
lambda = 0.05; % Factor de regularización (ajustable según el modelo)
nVars = size(outputData, 2); % Número de variables (vel_u, vel_v, vel_r)

% Construcción de la matriz de funciones Theta (biblioteca de funciones ampliada)
Theta = build_library(outputData, inputData, polyorder);

% Aplicación del algoritmo de regresión para encontrar el modelo disperso
Xi = sparsify_dynamics(Theta, outputDot, lambda);

% Visualizar los resultados
disp('Modelo identificado para el sistema dinámico (3DOF USV):');
model_equations = cell(nVars, 1); % Inicializar las ecuaciones del modelo

for i = 1:nVars
    fprintf('Ecuación para la derivada de la velocidad %d: ', i);
    model_equations{i} = display_equation(Xi(:, i), polyorder, i); % Guardar la expresión
end

% Mostrar la expresión completa del modelo
disp('Expresión completa del modelo:');
for i = 1:nVars
    fprintf('d(vel_%d)/dt = %s\n', i, model_equations{i});
end

% Funciones auxiliares
function Theta = build_library(outputData, inputData, polyorder)
    % Construye una biblioteca de funciones completa
    nData = size(outputData, 1);
    Theta = [ones(nData, 1)]; % Incluye el término constante
    nVars = size(outputData, 2);
    terms = [outputData, inputData];

    % Generar términos polinómicos de orden 1 y 2
    for i = 1:polyorder
        Theta = [Theta, terms.^i];
    end

    % Generar términos cruzados polinomiales (combinaciones entre variables)
    for i = 1:nVars
        for j = i:nVars
            Theta = [Theta, terms(:, i) .* terms(:, j)];
        end
    end

    % Añadir funciones trigonométricas
    Theta = [Theta, sin(terms), cos(terms)];
    
    % Añadir funciones exponenciales
    Theta = [Theta, exp(terms), exp(-terms)];
    
    % Términos logarítmicos (evitar log(0) al añadir un mínimo valor positivo)
    epsilon = 1e-6; % Pequeño valor para evitar log(0)
    Theta = [Theta, log(abs(terms) + epsilon)];
end

function Xi = sparsify_dynamics(Theta, dXdt, lambda)
    % Sparsify Dynamics - Selección de los términos dispersos en Theta
    [n, m] = size(Theta);
    Xi = zeros(m, size(dXdt, 2));
    
    for k = 1:size(dXdt, 2)
        Xi(:, k) = lasso(Theta, dXdt(:, k), 'Lambda', lambda);
    end
end

function equation = display_equation(Xi, polyorder, varIndex)
    % Muestra la ecuación para una derivada en base a los coeficientes Xi
    equation = ''; % Inicializar la cadena de la ecuación
    equation = [equation, sprintf('%.3f', Xi(1))]; % Término constante

    for i = 2:length(Xi)
        if Xi(i) ~= 0
            equation = [equation, sprintf(' + (%.3f) * X_%d', Xi(i), i)];
        end
    end
end
