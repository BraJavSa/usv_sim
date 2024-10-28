% Cargar los datos
load('ident_usv.mat'); % Asegúrate de que el archivo esté en la ruta actual

% Concatenar las entradas de torque
U = [T_u, T_r]; % Suponiendo que T_u y T_r son columnas de torques
V = [vel_u, vel_v, vel_r]; % Suponiendo que vel_u, vel_v y vel_r son las velocidades medidas

% Derivar las velocidades para obtener la dinámica
dV = gradient(V, ts); % Cálculo de derivadas

% Crear funciones polinómicas de las variables
poly_order = 3; % Orden del polinomio (puedes ajustarlo)
n_vars = size(V, 2);
n_torques = size(U, 2);
X = []; % Matriz de funciones

% Construir matriz de funciones
for i = 1:n_torques
    for j = 0:poly_order
        for k = 0:poly_order
            % Agregar todos los términos de las variables
            X = [X, (U(:, i)).^j .* (U(:, mod(i + k - 1, n_torques) + 1)).^k]; 
        end
    end
end

% Identificación SINDy
lambda = 0.1; % Parámetro de regularización (ajustable)
Theta = X; % Matriz de funciones

% Comprobar dimensiones
if size(dV, 2) ~= n_vars
    error('El número de columnas en dV no coincide con el número de variables en V.');
end

Xi = sparsifyDynamics(Theta, dV, lambda); % Función para identificar dinámicas

% Mostrar los resultados
disp('Coeficientes identificados por SINDy:');
disp(Xi);

% Función para sparsifyDynamics (SINDy)
function Xi = sparsifyDynamics(Theta, dV, lambda)
    % Mínimos cuadrados para ajustar el modelo
    [n_samples, n_vars] = size(Theta);
    Xi = zeros(size(Theta, 2), n_vars); % Inicializar matriz de coeficientes

    % Para cada variable en dV
    for i = 1:n_vars
        % Resolver el problema de mínimos cuadrados regularizado
        [B, ~] = lasso(Theta, dV(:, i), 'Lambda', lambda); % Lasso para sparsity
        Xi(:, i) = B; % Almacenar coeficientes
    end
end
