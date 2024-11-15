% Parámetros de la red neuronal
num_iteraciones = 3000;  % Número de iteraciones para generar datos
eta = 0.01;  % Tasa de aprendizaje
epoch = 500;  % Número máximo de épocas
umbral = 1e-4; % Umbral de error para la convergencia
num_neuronas = 1;  % Número de neuronas (salida)

% Inicialización de los datos (aquí asumo que los generas según tu código anterior)
pares_x = [];
pares_y = [];
claster = [];
for i = 1:num_iteraciones
    opcion = randi([1, 2]);
    if opcion == 1
        x = randi([0, 1000]) / 1000;  % Valores en el rango de 0 a 1
        y = 1 + randi([0, 1000]) / 1000; % Valores en el rango de 1 a 2
        pares_x = [pares_x, x];
        pares_y = [pares_y, y];
        claster = [claster, 1];
    else
        y = randi([0, 1000]) / 1000;  % Valores en el rango de 0 a 1
        x = 2 + randi([0, 1000]) / 1000; % Valores en el rango de 1 a 2
        pares_x = [pares_x, x];
        pares_y = [pares_y, y];
        claster = [claster, 0];
    end
end

% Concatenamos las entradas X y Y en una matriz de entradas
entradas = [pares_x; pares_y]';  % Matriz de entradas, tamaño Nx2
salidas = claster';  % Vectores de salida (clase 1 o 0)

% Inicialización de los pesos y el sesgo
pesos = rand(1, num_neuronas) * 0.1;  % Inicialización aleatoria pequeña
sesgo = 0;  % Inicialización del sesgo en cero

% Entrenamiento de la red neuronal
for epoca = 1:epoch
    error_total = 0;  % Resetear el error total en cada época

    for i = 1:num_iteraciones
        % Calcular la salida de la neurona (función de activación: identidad)
        salida_neuronal = entradas(i, :) * pesos' + sesgo;  % Producto punto + sesgo

        % Calcular el error (diferencia entre la salida deseada y la salida de la red)
        error = salidas(i) - salida_neuronal;  % Error de la red

        % Actualización de los pesos y el sesgo utilizando la regla delta
        pesos = pesos + eta * error .* entradas(i, :);  % Actualización de los pesos
        sesgo = sesgo + eta * error;  % Actualización del sesgo

        % Sumar el error total (para monitorear la convergencia)
        error_total = error_total + sum(error.^2);
    end

    % Mostrar el error total en cada época
    disp(['Epoch ' num2str(epoca) ', Error total: ' num2str(error_total)]);

    % Condición de convergencia
    if error_total < umbral
        disp('Red neuronal convergió');
        break;
    end
end

% Predicciones finales con los pesos aprendidos
predicciones = entradas * pesos' + sesgo;  % Salida de la red para cada entrada
predicciones_clase = round(predicciones);  % Redondeamos a 0 o 1 para clasificar

% Graficar los resultados
figure;
hold on;

% Predicciones de la clase 1 (rojo)
scatter(pares_x(predicciones_clase == 1), pares_y(predicciones_clase == 1), 'r', 'filled');

% Predicciones de la clase 0 (azul)
scatter(pares_x(predicciones_clase == 0), pares_y(predicciones_clase == 0), 'b', 'filled');

% Puntos reales de la clase 1
scatter(pares_x(claster == 1), pares_y(claster == 1), 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');

% Puntos reales de la clase 0
scatter(pares_x(claster == 0), pares_y(claster == 0), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');

legend('Predicción clase 1', 'Predicción clase 0', 'Clase real 1', 'Clase real 0');
xlabel('X');
ylabel('Y');
title('Clasificación de puntos con la red neuronal');
grid on;
hold off;
