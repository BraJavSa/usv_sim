% Rango de comandos de empuje en porcentaje
cmd_values = -100:10:100;  % Ahora va de -100% a 100%

% Inicializar vector para almacenar los valores de la fuerza
force_values = zeros(size(cmd_values));

% Calcular la fuerza para cada valor de cmd
for i = 1:length(cmd_values)
    force_values(i) = calcularFuerzaPropulsor(cmd_values(i) / 100);  % Dividir por 100 para usar en la función
end

% Configurar la gráfica con un tamaño rectangular
figure('Color', 'w', 'Position', [100, 100, 800, 400]); % Fondo blanco y tamaño rectangular
scatter(cmd_values, force_values, 60, 'filled', 'MarkerFaceColor', [0.5, 0.5, 0.5], 'MarkerEdgeColor', 'k');
hold on;

% Títulos y etiquetas de los ejes con Times New Roman y sin negrita
xlabel('% Motor Command', 'FontSize', 12, 'FontName', 'Times New Roman', 'HorizontalAlignment', 'center'); % Centrado en la parte inferior
ylabel('Thruster Force [N]', 'FontSize', 12, 'FontName', 'Times New Roman');  % Mantener en posición vertical original

% Ajuste de los ejes y mover al centro
grid on;
set(gca, 'GridLineStyle', '--', 'GridColor', 'k', 'GridAlpha', 0.5);
xlim([-100 100]);
ylim([-120 120]);

% Guardar como archivo PDF
saveas(gcf, 'ThrusterForceGraph.pdf');
