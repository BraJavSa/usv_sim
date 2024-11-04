%% Clear Variables
clc, clear all, close all;
clear tf;

%% LOAD VALUES FROM MATRICES
load('ident_usv.mat')
load('usv_data2.mat')

% Variables de frecuencia de muestreo y parámetros
ts = 1 / 50; % Frecuencia de muestreo 50 Hz
landa = 100000; % Parámetro de filtrado

% Definir filtro de primer orden para suavizar las señales
F1 = tf(landa, [1 landa]);

% Aplicar el filtro a las señales de velocidad y fuerza
u_f = lsim(F1, vel_u, t)';
v_f = lsim(F1, vel_v, t)';
r_f = lsim(F1, vel_r, t)';

T_u_f = lsim(F1, T_u, t)';
T_r_f = lsim(F1, T_r, t)';

% Agrupar las entradas y salidas filtradas
inputs = [T_u_f; T_r_f];  % Entradas de fuerza (filtradas)
outputs = [u_f; v_f; r_f];  % Velocidades (filtradas)

%% Normalización de datos
inputMean = mean(inputs, 2);
inputStd = std(inputs, 0, 2);
inputsNorm = (inputs - inputMean) ./ inputStd;

outputMean = mean(outputs, 2);
outputStd = std(outputs, 0, 2);
outputsNorm = (outputs - outputMean) ./ outputStd;

%% División de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
numTimeSteps = size(inputsNorm, 2);
numTrain = round(0.8 * numTimeSteps);

XTrain = inputsNorm(:, 1:numTrain);
YTrain = outputsNorm(:, 1:numTrain);
XTest = inputsNorm(:, numTrain+1:end);
YTest = outputsNorm(:, numTrain+1:end);

%% Configuración de la Red LSTM para Predicción de Velocidades
inputSize = size(XTrain, 1); % Dimensión de entrada (2: T_u y T_r)
numHiddenUnits = 400; % Número de unidades ocultas, reducido para optimización
outputSize = size(YTrain, 1); % Dimensión de salida (3: vel_u, vel_v, vel_r)

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(outputSize)
    regressionLayer];

%% Opciones de entrenamiento
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 10, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% Entrenar la Red LSTM
net = trainNetwork(XTrain, YTrain, layers, options);

%% Predicción y Desnormalización
YPredNorm = predict(net, XTest, 'MiniBatchSize', 1);
YPred = YPredNorm .* outputStd + outputMean;  % Desnormalizar predicción
YTestReal = YTest .* outputStd + outputMean;  % Desnormalizar valores de prueba

%% Visualización de Resultados
timeTest = t(numTrain+1:end); % Tiempo para datos de prueba

figure;
subplot(3, 1, 1);
plot(timeTest, YTestReal(1, :), 'r', timeTest, YPred(1, :), 'b');
title('Comparación de velocidad u'); xlabel('Time (s)'); ylabel('vel\_u');
legend('Real', 'Predicción');

subplot(3, 1, 2);
plot(timeTest, YTestReal(2, :), 'r', timeTest, YPred(2, :), 'b');
title('Comparación de velocidad v'); xlabel('Time (s)'); ylabel('vel\_v');
legend('Real', 'Predicción');

subplot(3, 1, 3);
plot(timeTest, YTestReal(3, :), 'r', timeTest, YPred(3, :), 'b');
title('Comparación de velocidad r'); xlabel('Time (s)'); ylabel('vel\_r');
legend('Real', 'Predicción');
