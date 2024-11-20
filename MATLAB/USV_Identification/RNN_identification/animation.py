import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import loadmat
import torch
import torch.nn as nn

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

# Load data
data_val = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/muestreo_externo_2.mat')
val_dataset, T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val, vel_u_val, vel_v_val, vel_r_val = create_dataset(data_val)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define model parameters
input_size = 8  # Input features: T_u, T_r, pass_vel_u, pass_vel_v, pass_vel_r, pass2_vel_u, pass2_vel_v, pass2_vel_r
hidden_size = 200
output_size = 3  # Output features: vel_u, vel_v, vel_r

# Load the model
model = USVRNN(input_size, hidden_size, output_size)
model_path = '/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/usv_rnn_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
print("Modelo cargado y listo para validación")

# Initialize empty arrays for predictions and real values
vel_u_pred_all = []
vel_v_pred_all = []
vel_r_pred_all = []
vel_u_val_all = []
vel_v_val_all = []
vel_r_val_all = []

ts = 0.02  # Sample time
time_steps = []

fig, ax = plt.subplots(3, 1, figsize=(12, 8))

lines = []
for i in range(3):
    line, = ax[i].plot([], [], label='Real')
    pred_line, = ax[i].plot([], [], label='Predicha')
    lines.append(line)
    lines.append(pred_line)
    ax[i].set_xlim(0, 1)
    ax[i].set_ylim(-1, 1)
    ax[i].legend()

ax[0].set_ylabel('Velocidad de Surgencia (m/s)')
ax[1].set_ylabel('Velocidad de Balanceo (m/s)')
ax[2].set_ylabel('Velocidad de Guiñada (rad/s)')
ax[2].set_xlabel('Tiempo (s)')

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    global time_steps, vel_u_pred_all, vel_v_pred_all, vel_r_pred_all, vel_u_val_all, vel_v_val_all, vel_r_val_all

    T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val, vel_u_val, vel_v_val, vel_r_val = frame
    input_tensor_val = torch.stack([T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val], dim=1).unsqueeze(0)

    with torch.no_grad():
        output_val = model(input_tensor_val)
    
    vel_u_pred = output_val[:, 0].numpy().flatten()
    vel_v_pred = output_val[:, 1].numpy().flatten()
    vel_r_pred = output_val[:, 2].numpy().flatten()

    time_steps.append(time_steps[-1] + ts if time_steps else 0)
    vel_u_pred_all.append(vel_u_pred[0])
    vel_v_pred_all.append(vel_v_pred[0])
    vel_r_pred_all.append(vel_r_pred[0])
    vel_u_val_all.append(vel_u_val.item())
    vel_v_val_all.append(vel_v_val.item())
    vel_r_val_all.append(vel_r_val.item())

    lines[0].set_data(time_steps, vel_u_val_all)
    lines[1].set_data(time_steps, vel_u_pred_all)
    lines[2].set_data(time_steps, vel_v_val_all)
    lines[3].set_data(time_steps, vel_v_pred_all)
    lines[4].set_data(time_steps, vel_r_val_all)
    lines[5].set_data(time_steps, vel_r_pred_all)

    for ax_i in ax:
        ax_i.set_xlim(0, max(1, time_steps[-1]))
        ax_i.set_ylim(min(min(vel_u_val_all), min(vel_u_pred_all), -1), max(max(vel_u_val_all), max(vel_u_pred_all), 1))

    return lines

frames = list(val_dataloader)

ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False)
plt.tight_layout()
plt.show()
