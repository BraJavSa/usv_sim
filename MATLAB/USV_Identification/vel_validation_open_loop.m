
load('torque_velocity_data.mat')

% Inicializar un vector para almacenar los promedios
promedios_cambio = [];
valores = [];
% Inicializar la posición inicial para el promedio
inicio = 1;

% Recorrer el vector y detectar cambios
for k = 2:length(left_data)
    if left_data(k) ~= left_data(k-1) % Verificar si hay un cambio en el valor
        % Calcular el promedio de vector2 entre 'inicio' y 'k'
        promedio = mean(linear_velocity_data(inicio:k-1));
        promedios_cambio = [promedios_cambio, promedio]; % Almacenar el promedio
        valores= [valores, left_data(k-1)]; % Almacenar el promedio
        inicio = k; % Actualizar la posición inicial para el siguiente rango
    end
end

% Calcular el promedio para el último segmento, si hay cambios
if inicio <= length(left_data)
    promedio = mean(linear_velocity_data(inicio:end));
    promedios_cambio = [promedios_cambio, promedio]; % Almacenar el promedio del último segmento
    valores= [valores, left_data(10208)]; % Almacenar el promedio
end

% Tomar los primeros 20 valores de cada vector
x = [promedios_cambio(13:21), promedios_cambio(2:9)];
y = [valores(13:21), valores(2:9)];
% Graficar los valores
figure; % Crear una nueva figura
plot(x,y, 'o-', 'LineWidth', 2, 'MarkerSize', 8); % Graficar con marcadores y líneas
grid on; % Activar la cuadrícula
title('Gráfica de los Primeros 20 Valores de los Vectores');
xlabel('Primer Vector (vector1)');
ylabel('Segundo Vector (vector2)');
xlim([min(x)-0.1, max(x)+0.1]); % Ajustar los límites del eje x
ylim([min(y)-0.1, max(y)+0.1]); % Ajustar los límites del eje y
legend('Valores de vector2 en función de vector1', 'Location', 'best'); % Añadir leyenda