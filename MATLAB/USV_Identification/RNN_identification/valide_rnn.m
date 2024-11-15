% Import the ONNX model as a function
modelFunction = importONNXFunction('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/Sindy_Identification/usv_rnn.onnx', 'usv_rnn');

% Load the validation data
data_val = load('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/Sindy_Identification/muestreo_externo.mat');
t_val = data_val.t;
T_u_val = data_val.T_u;
T_r_val = data_val.T_r;
vel_u_val = data_val.vel_u;
vel_v_val = data_val.vel_v;
vel_r_val = data_val.vel_r;

% Prepare the input data
input_data = cat(3, T_u_val, T_r_val); % Combine T_u and T_r along the third dimension

% Predict the outputs using the imported ONNX function
predictions = usv_rnn(input_data);

% Extract predicted velocities
vel_u_pred = squeeze(predictions(:, 1, :));
vel_v_pred = squeeze(predictions(:, 2, :));
vel_r_pred = squeeze(predictions(:, 3, :));

% Plot the results
figure;
subplot(3, 1, 1);
plot(t_val, vel_u_val, 'b', 'DisplayName', 'Velocidad de Surgencia Real');
hold on;
plot(t_val, vel_u_pred, 'r', 'DisplayName', 'Velocidad de Surgencia Predicha');
xlabel('Tiempo (s)');
ylabel('Velocidad de Surgencia (m/s)');
legend;
hold off;

subplot(3, 1, 2);
plot(t_val, vel_v_val, 'b', 'DisplayName', 'Velocidad de Balanceo Real');
hold on;
plot(t_val, vel_v_pred, 'r', 'DisplayName', 'Velocidad de Balanceo Predicha');
xlabel('Tiempo (s)');
ylabel('Velocidad de Balanceo (m/s)');
legend;
hold off;

subplot(3, 1, 3);
plot(t_val, vel_r_val, 'b', 'DisplayName', 'Velocidad de Guiñada Real');
hold on;
plot(t_val, vel_r_pred, 'r', 'DisplayName', 'Velocidad de Guiñada Predicha');
xlabel('Tiempo (s)');
ylabel('Velocidad de Guiñada (rad/s)');
legend;
hold off;

sgtitle('Evolución de las Velocidades Predichas vs Reales');
