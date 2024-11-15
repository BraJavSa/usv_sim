% Cargar los datos del archivo .mat
data = load('torque_identification.mat');

% Extraer las variables necesarias
left_cmd = data.left_cmd; % Comando de torque izquierdo
linear_velocity_data = data.linear_velocity_data; % Velocidades lineales

% Calcular la línea de ajuste lineal
p = polyfit(left_cmd, linear_velocity_data, 1); % Ajuste lineal (grado 1)
yfit = polyval(p, left_cmd); % Valores ajustados

% Crear el gráfico
figure; % Crear una nueva figura
plot(left_cmd, linear_velocity_data, 'o', 'MarkerFaceColor', 'b'); % Datos originales como puntos
hold on; % Mantener el gráfico
plot(left_cmd, yfit, 'r-', 'LineWidth', 2); % Graficar la línea de ajuste
xlabel('Torque Izquierdo (left_cmd)'); % Etiqueta del eje X
ylabel('Velocidad Lineal (linear_velocity_data)'); % Etiqueta del eje Y
title('Relación entre Torque Izquierdo y Velocidad Lineal con Ajuste Lineal'); % Título del gráfico
legend('Datos', 'Línea de Ajuste', 'Location', 'best'); % Leyenda
grid on; % Activar la cuadrícula
hold off; % Liberar el gráfico