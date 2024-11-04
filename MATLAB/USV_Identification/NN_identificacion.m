% Load data
load('ident_usv.mat');  % Assumes variables: vel_u, vel_v, vel_r, T_u, T_r

% Prepare data for RNN
% Convert each time series to cell arrays with one time step per cell
inputSeq = mat2cell([T_u, T_r], ones(11500, 1), 2); % Inputs: 11500 timesteps, 2 inputs per timestep
outputSeq = mat2cell([vel_u, vel_v, vel_r], ones(11500, 1), 3); % Outputs: 11500 timesteps, 3 outputs per timestep

% Define RNN Architecture with ReLU Activation
inputSize = 2;          % Two inputs: T_u and T_r
outputSize = 3;         % Three outputs: vel_u, vel_v, and vel_r
numHiddenUnits = 100;   % Number of hidden units for capturing dynamics

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    reluLayer               % Adding ReLU activation layer after the LSTM layer
    fullyConnectedLayer(outputSize)
    regressionLayer];

% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.005, ...
    'GradientThreshold', 1, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% Train the Network
net = trainNetwork(inputSeq, outputSeq, layers, options);

% Test the Network on Training Data (for validation)
predictedOutputSeq = predict(net, inputSeq);

% Convert cell arrays to matrices for easier comparison
predictedOutputMat = cell2mat(predictedOutputSeq);
outputMat = cell2mat(outputSeq);

% Evaluate performance (e.g., MSE, correlation)
mseError = mean((predictedOutputMat - outputMat).^2, 'all');
fprintf('Mean Squared Error: %f\n', mseError);

% Plot results for each output
figure;
subplot(3,1,1);
plot(outputMat(:,1), 'b'); hold on;
plot(predictedOutputMat(:,1), 'r--');
title('Surge Velocity u (Actual vs. Predicted)'); legend('Actual', 'Predicted');

subplot(3,1,2);
plot(outputMat(:,2), 'b'); hold on;
plot(predictedOutputMat(:,2), 'r--');
title('Sway Velocity v (Actual vs. Predicted)'); legend('Actual', 'Predicted');

subplot(3,1,3);
plot(outputMat(:,3), 'b'); hold on;
plot(predictedOutputMat(:,3), 'r--');
title('Yaw Rate r (Actual vs. Predicted)'); legend('Actual', 'Predicted');
