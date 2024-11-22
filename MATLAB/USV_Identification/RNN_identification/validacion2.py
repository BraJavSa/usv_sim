import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lifelines.utils import concordance_index


class USVDataset(torch.utils.data.Dataset):
    def __init__(self, T_u, T_r, pass_vel_u, pass_vel_v, pass_vel_r, pass2_vel_u, pass2_vel_v, pass2_vel_r, vel_u, vel_v, vel_r):
        self.T_u = torch.tensor(T_u, dtype=torch.float32)
        self.T_r = torch.tensor(T_r, dtype=torch.float32)
        self.pass_vel_u = torch.tensor(pass_vel_u, dtype=torch.float32)
        self.pass_vel_v = torch.tensor(pass_vel_v, dtype=torch.float32)
        self.pass_vel_r = torch.tensor(pass_vel_r, dtype=torch.float32)
        self.pass2_vel_u = torch.tensor(pass2_vel_u, dtype=torch.float32)
        self.pass2_vel_v = torch.tensor(pass2_vel_v, dtype=torch.float32)
        self.pass2_vel_r = torch.tensor(pass2_vel_r, dtype=torch.float32)
        self.vel_u = torch.tensor(vel_u, dtype=torch.float32)
        self.vel_v = torch.tensor(vel_v, dtype=torch.float32)
        self.vel_r = torch.tensor(vel_r, dtype=torch.float32)

    def __len__(self):
        return len(self.T_u)

    def __getitem__(self, idx):
        return (self.T_u[idx], self.T_r[idx], self.pass_vel_u[idx], self.pass_vel_v[idx], self.pass_vel_r[idx],
                self.pass2_vel_u[idx], self.pass2_vel_v[idx], self.pass2_vel_r[idx],
                self.vel_u[idx], self.vel_v[idx], self.vel_r[idx])

class USVRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(USVRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Clase para el modelo MLP
class USVNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(USVNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the create_dataset function
def create_dataset(data):
    T_u = data['T_u'].flatten()
    T_r = data['T_r'].flatten()
    pass_vel_u = data['pass_vel_u'].flatten()
    pass_vel_v = data['pass_vel_v'].flatten()
    pass_vel_r = data['pass_vel_r'].flatten()
    pass2_vel_u = data['pass2_vel_u'].flatten()
    pass2_vel_v = data['pass2_vel_v'].flatten()
    pass2_vel_r = data['pass2_vel_r'].flatten()
    vel_u = data['vel_u'].flatten()
    vel_v = data['vel_v'].flatten()
    vel_r = data['vel_r'].flatten()

    dataset = USVDataset(T_u, T_r, pass_vel_u, pass_vel_v, pass_vel_r, pass2_vel_u, pass2_vel_v, pass2_vel_r, vel_u, vel_v, vel_r)
    return dataset, T_u, T_r, pass_vel_u, pass_vel_v, pass_vel_r, pass2_vel_u, pass2_vel_v, pass2_vel_r, vel_u, vel_v, vel_r

# Definición de matrices del sistema
def define_matrices(delta):
    M = np.array([[delta[0], 0, 0],
                  [0, delta[1], delta[2]],
                  [0, delta[2], delta[3]]])
    D = np.array([[delta[7], 0, 0],
                  [0, delta[8], delta[9]],
                  [0, delta[9], delta[10]]])
    return M, D

# Función para cargar y preprocesar datos desde archivos .mat
def load_data(mat_file):
    data = loadmat(mat_file)
    return data

# Función para la simulación LSQ
def simulate_lsq(T_u, T_r, vel_u, vel_v, vel_r, delta, t, ts):
    M, D = define_matrices(delta)
    IM = np.linalg.inv(M)
    u = np.zeros(len(t))
    v = np.zeros(len(t))
    r = np.zeros(len(t))
    T_v = np.zeros(len(t))  # Asumiendo T_v = 0

    for k in range(len(t) - 1):
        if k >= len(T_u) or k >= len(T_r):
            break
        vel = np.array([u[k], v[k], r[k]])
        C = np.array([[0, -delta[4] * r[k], -delta[5] * v[k] - delta[2] * r[k]],
                      [delta[4] * r[k], 0, delta[6] * u[k]],
                      [delta[5] * v[k] + delta[2] * r[k], -delta[6] * u[k], 0]])
        d_vel = IM @ (np.array([T_u[k], T_v[k], T_r[k]]) - C @ vel - D @ vel)
        u[k + 1] = u[k] + d_vel[0] * ts
        v[k + 1] = v[k] + d_vel[1] * ts
        r[k + 1] = r[k] + d_vel[2] * ts

    return u, v, r
# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to calculate MAE
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Function to calculate Concordance Index
def calculate_cindex(y_true, y_pred):
    return concordance_index(y_true, y_pred)
# Load data
data_val = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/muestreo_externo_4.mat')
val_dataset, T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val, vel_u_val, vel_v_val, vel_r_val = create_dataset(data_val)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model parameters
input_size = 8  # Input features: T_u, T_r, pass_vel_u, pass_vel_v, pass_vel_r, pass2_vel_u, pass2_vel_v, pass2_vel_r
hidden_size = 200
output_size = 3  # Output features: vel_u, vel_v, vel_r

# Load the RNN model
model_rnn = USVRNN(input_size, hidden_size, output_size)
model_path_rnn = '/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/usv_rnn_model_2.pth'
model_rnn.load_state_dict(torch.load(model_path_rnn))
model_rnn.eval()
print("Modelo RNN cargado y listo para validación")

# Load the MLP model
model_mlp = USVNN(input_size, hidden_size, output_size)
model_path_mlp = '/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/usv_nn_model.pth'
model_mlp.load_state_dict(torch.load(model_path_mlp))
model_mlp.eval()
print("Modelo MLP cargado y listo para validación")

# Initialize empty arrays for predictions and real values
vel_u_pred_rnn = []
vel_v_pred_rnn = []
vel_r_pred_rnn = []
vel_u_pred_mlp = []
vel_v_pred_mlp = []
vel_r_pred_mlp = []
T_u_val_all = []
T_r_val_all = []
pass_vel_u_val_all = []
pass_vel_v_val_all = []
pass_vel_r_val_all = []
pass2_vel_u_val_all = []
pass2_vel_v_val_all = []
pass2_vel_r_val_all = []
vel_u_val_all = []
vel_v_val_all = []
vel_r_val_all = []

with torch.no_grad():
    for (T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val,
         vel_u_val, vel_v_val, vel_r_val) in val_dataloader:
        input_tensor_rnn = torch.stack([T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val], dim=1).unsqueeze(1)
        # Perform prediction with RNN
        output_rnn = model_rnn(input_tensor_rnn)
        # Extract predicted velocities from RNN
        vel_u_pred_rnn.extend(output_rnn[:, 0].numpy().flatten())
        vel_v_pred_rnn.extend(output_rnn[:, 1].numpy().flatten())
        vel_r_pred_rnn.extend(output_rnn[:, 2].numpy().flatten())
        
        # Prepare input tensor for MLP
        input_tensor_mlp = torch.stack([T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val], dim=1)
        # Perform prediction with MLP
        output_mlp = model_mlp(input_tensor_mlp)
        # Extract predicted velocities from MLP
        vel_u_pred_mlp.extend(output_mlp[:, 0].numpy().flatten())
        vel_v_pred_mlp.extend(output_mlp[:, 1].numpy().flatten())
        vel_r_pred_mlp.extend(output_mlp[:, 2].numpy().flatten())

        # Store real values
        T_u_val_all.extend(T_u_val.numpy().flatten())
        T_r_val_all.extend(T_r_val.numpy().flatten())
        pass_vel_u_val_all.extend(pass_vel_u_val.numpy().flatten())
        pass_vel_v_val_all.extend(pass_vel_v_val.numpy().flatten())
        pass_vel_r_val_all.extend(pass_vel_r_val.numpy().flatten())
        pass2_vel_u_val_all.extend(pass2_vel_u_val.numpy().flatten())
        pass2_vel_v_val_all.extend(pass2_vel_v_val.numpy().flatten())
        pass2_vel_r_val_all.extend(pass2_vel_r_val.numpy().flatten())
        vel_u_val_all.extend(vel_u_val.numpy().flatten())
        vel_v_val_all.extend(vel_v_val.numpy().flatten())
        vel_r_val_all.extend(vel_r_val.numpy().flatten())

delta_values = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/delta_valores4_1.mat')['delta'].flatten()
#delta_values[0] = 96.2918752627172

t = data_val['t'].flatten()
ts = t[1] - t[0]  # Asumiendo un tiempo de muestreo constante

# Ejecutar la simulación LSQ
u_lsq, v_lsq, r_lsq = simulate_lsq(data_val['T_u'].flatten(), data_val['T_r'].flatten(),
                                   data_val['vel_u'].flatten(), data_val['vel_v'].flatten(),
                                   data_val['vel_r'].flatten(), delta_values, t, ts)

# Convert lists to numpy arrays
vel_u_pred_rnn = np.array(vel_u_pred_rnn)
vel_v_pred_rnn = np.array(vel_v_pred_rnn)
vel_r_pred_rnn = np.array(vel_r_pred_rnn)
vel_u_pred_mlp = np.array(vel_u_pred_mlp)
vel_v_pred_mlp = np.array(vel_v_pred_mlp)
vel_r_pred_mlp = np.array(vel_r_pred_mlp)
T_u_val_all = np.array(T_u_val_all)
T_r_val_all = np.array(T_r_val_all)
pass_vel_u_val_all = np.array(pass_vel_u_val_all)
pass_vel_v_val_all = np.array(pass_vel_v_val_all)
pass_vel_r_val_all = np.array(pass_vel_r_val_all)
pass2_vel_u_val_all = np.array(pass2_vel_u_val_all)
pass2_vel_v_val_all = np.array(pass2_vel_v_val_all)
pass2_vel_r_val_all = np.array(pass2_vel_r_val_all)
vel_u_val_all = np.array(vel_u_val_all)
vel_v_val_all = np.array(vel_v_val_all)
vel_r_val_all = np.array(vel_r_val_all)


time_steps = np.arange(0, len(vel_u_val_all) * 0.02, 0.02)

# Actual and predicted values for each velocity component
velocities = {
    'u': {'actual': vel_u_val_all, 'pred_rnn': vel_u_pred_rnn, 'pred_mlp': vel_u_pred_mlp, 'pred_lsq': u_lsq[:len(time_steps)]},
    'v': {'actual': vel_v_val_all, 'pred_rnn': vel_v_pred_rnn, 'pred_mlp': vel_v_pred_mlp, 'pred_lsq': v_lsq[:len(time_steps)]},
    'r': {'actual': vel_r_val_all, 'pred_rnn': vel_r_pred_rnn, 'pred_mlp': vel_r_pred_mlp, 'pred_lsq': r_lsq[:len(time_steps)]}
}

metrics = {}

for key, value in velocities.items():
    metrics[key] = {
        'RMSE_RNN': calculate_rmse(value['actual'], value['pred_rnn']),
        'MAE_RNN': calculate_mae(value['actual'], value['pred_rnn']),
        'CIndex_RNN': calculate_cindex(value['actual'], value['pred_rnn']),
        'RMSE_MLP': calculate_rmse(value['actual'], value['pred_mlp']),
        'MAE_MLP': calculate_mae(value['actual'], value['pred_mlp']),
        'CIndex_MLP': calculate_cindex(value['actual'], value['pred_mlp']),
        'RMSE_LSQ': calculate_rmse(value['actual'], value['pred_lsq']),
        'MAE_LSQ': calculate_mae(value['actual'], value['pred_lsq']),
        'CIndex_LSQ': calculate_cindex(value['actual'], value['pred_lsq']),
    }

# Print results
for vel, met in metrics.items():
    print(f"Metrics for velocity {vel.upper()}:")
    for metric, value in met.items():
        print(f"{metric}: {value}")
    print("\n")


# Plot the results for velocities
plt.figure(figsize=(12, 16))
# Crear un nuevo vector de tiempo basado en el muestreo de 0.02 segundos

# Subplot for surge velocity u
plt.subplot(3, 1, 1)
plt.plot(time_steps, vel_u_val_all, label='Real Surge Velocity')
plt.plot(time_steps, vel_u_pred_rnn, label='Predicted Surge Velocity (RNN)')
plt.plot(time_steps, vel_u_pred_mlp, label='Predicted Surge Velocity (MLP)')
plt.plot(time_steps, u_lsq[:len(time_steps)], label='Predicted Surge Velocity (LSQ)')
plt.xlim(0, 30) # Ajustar el límite del eje X
plt.ylabel('Surge Velocity (m/s)')
plt.legend()
plt.grid(True)

# Subplot for sway velocity v
plt.subplot(3, 1, 2)
plt.plot(time_steps, vel_v_val_all, label='Real Sway Velocity')
plt.plot(time_steps, vel_v_pred_rnn, label='Predicted Sway Velocity (RNN)')
plt.plot(time_steps, vel_v_pred_mlp, label='Predicted Sway Velocity (MLP)')
plt.plot(time_steps, v_lsq[:len(time_steps)], label='Predicted Sway Velocity (LSQ)')
plt.xlim(0, 30) # Ajustar el límite del eje X
plt.ylabel('Sway Velocity (m/s)')
plt.legend()
plt.grid(True)

# Subplot for yaw velocity r
plt.subplot(3, 1, 3)
plt.plot(time_steps, vel_r_val_all, label='Real Yaw Velocity')
plt.plot(time_steps, vel_r_pred_rnn, label='Predicted Yaw Velocity (RNN)')
plt.plot(time_steps, vel_r_pred_mlp, label='Predicted Yaw Velocity (MLP)')
plt.plot(time_steps, r_lsq[:len(time_steps)], label='Predicted Yaw Velocity (LSQ)')
plt.xlim(0, 30) # Ajustar el límite del eje X
plt.xlabel('Time (seconds)')
plt.ylabel('Yaw Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
