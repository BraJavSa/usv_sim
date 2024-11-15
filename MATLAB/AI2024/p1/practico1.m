%%creando claster
% Número de iteraciones
num_iteraciones = 6000;

% Almacenar coordenadas pares generadas
pares_x = [];
pares_y = [];
claster = [];
% Iterar 3000 veces para generar números aleatorios pares en el rectángulo
for i = 1:num_iteraciones
    
      opcion = randi([1, 2]);

    % Usar un if para actuar según la opción generada
    if opcion == 1
        % Generar un número aleatorio par en el rango [0, 1] para X y [1, 2] para Y
        x = randi([0, 1000]) / 1000;  % Valores en el rango de 0 a 1
        y = 1 + randi([0, 1000]) / 1000; % Valores en el rango de 1 a 2
        pares_x = [pares_x, x];
        pares_y = [pares_y, y];
        claster =[claster, 1];
    else
        % Generar un número aleatorio par en el rango [0, 1] para X y [1, 2] para Y
        y = randi([0, 1000]) / 1000;  % Valores en el rango de 0 a 1
        x = 2 + randi([0, 1000]) / 1000; % Valores en el rango de 1 a 2
        pares_x = [pares_x, x];
        pares_y = [pares_y, y];
        claster =[claster, 0];

    end

end



% Parámetros
learning_rate = 0.1; % Tasa de aprendizaje
epochs = 100; % Número de épocas de entrenamiento

% Inicialización de los parámetros de la neurona
w = rand(1, 2); % Pesos iniciales aleatorios para las dos entradas (x e y)
b = rand; % Sesgo inicial aleatorio

% Convertir los datos en un formato adecuado
inputs = [pares_x; pares_y]';

% Entrenamiento de la red neuronal
for epoch = 1:epochs
    for i = 1:length(claster)
        % Cálculo de la salida de la neurona (activación lineal)
        output = dot(w, inputs(i, :)) + b;
        
        % Función de activación (escalón)
        predicted_label = output >= 0;
        
        % Error (diferencia entre la predicción y la etiqueta real)
        error = claster(i) - predicted_label;
        
        % Actualización de los pesos y el sesgo
        w = w + learning_rate * error * inputs(i, :);
        b = b + learning_rate * error;
    end
end

% Graficar los puntos generados y la línea de separación en una sola gráfica
figure;
hold on;

% Graficar los puntos generados
scatter(pares_x(claster == 1), pares_y(claster == 1), 'r', 'filled'); % Clúster 1 en rojo
scatter(pares_x(claster == 0), pares_y(claster == 0), 'b', 'filled'); % Clúster 2 en azul

% Graficar la línea de separación calculada por la red neuronal
x_vals = linspace(min(pares_x) - 0.5, max(pares_x) + 0.5, 100);
y_vals = -(w(1) * x_vals + b) / w(2); % y = -(w1/w2)*x - b/w2
plot(x_vals, y_vals, 'k--', 'LineWidth', 2); % Línea de separación en negro punteado

% Configuración de la gráfica
xlabel('X');
ylabel('Y');
title('Clasificación con una Neurona Simple');
legend('Clúster 1', 'Clúster 2', 'Línea de separación');
xlim([0, 3]);
ylim([0, 3]);
grid on;
hold off;

