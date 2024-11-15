import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class USVDataset(torch.utils.data.Dataset):
    def __init__(self, t, vel_u, vel_v, vel_r, T_u, T_r):
        self.t = torch.tensor(t, dtype=torch.float32)
        self.vel_u = torch.tensor(vel_u, dtype=torch.float32)
        self.vel_v = torch.tensor(vel_v, dtype=torch.float32)
        self.vel_r = torch.tensor(vel_r, dtype=torch.float32)
        self.T_u = torch.tensor(T_u, dtype=torch.float32)
        self.T_r = torch.tensor(T_r, dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.vel_u[idx], self.vel_v[idx], self.vel_r[idx], self.T_u[idx], self.T_r[idx]

# Create the RNN model
class USVRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(USVRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:,-1, :])
        return x

# Extract features and create datasets
def create_dataset(data):
    t = data['t'].flatten()
    vel_u = data['vel_u'].flatten()
    vel_v = data['vel_v'].flatten()
    vel_r = data['vel_r'].flatten()
    T_u = data['T_u'].flatten()
    T_r = data['T_r'].flatten()

    dataset = USVDataset(t, vel_u, vel_v, vel_r, T_u, T_r)
    return dataset

# Load data
data = loadmat('ident_usv.mat')
data_val = loadmat('muestreo_externo.mat')



train_dataset = create_dataset(data)
val_dataset = create_dataset(data_val)


# Hyperparameters
input_size = 6  # Input features: t, vel_u, vel_v, vel_r, T_u, T_r
hidden_size = 20
output_size = 3  # Output features: d(vel_u)/dt, d(vel_v)/dt, d(vel_r)/dt
learning_rate = 0.01
num_epochs = 100

# Create the model, loss function, and optimizer
model = USVRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the dataset and data loader
dataset = USVDataset(t, vel_u, vel_v, vel_r, T_u, T_r)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (t, vel_u, vel_v, vel_r, T_u, T_r) in enumerate(dataloader):
        # Prepare input and target tensors
        input_tensor = torch.stack([t, vel_u, vel_v, vel_r, T_u, T_r], dim=1).unsqueeze(0)
        target_tensor = torch.stack([torch.diff(vel_u), torch.diff(vel_v), torch.diff(vel_r)], dim=1)

        # Forward pass
        output = model(input_tensor)

        # Calculate loss
        loss = criterion(output, target_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# After training,

# Evaluar el modelo
model.eval()

# Crear listas para almacenar los resultados
vel_u_pred_all = []
vel_v_pred_all = []
vel_r_pred_all = []

with torch.no_grad():
    for t_val, vel_u_val, vel_v_val, vel_r_val, T_u_val, T_r_val in val_dataloader:
        # Preparar el tensor de entrada
        input_tensor_val = torch.stack([t_val, vel_u_val, vel_v_val, vel_r_val, T_u_val, T_r_val], dim=1).unsqueeze(0)

        # Realizar la predicción
        output_val = model(input_tensor_val)

        # Calcular las velocidades predichas
        vel_u_pred = vel_u_val + output_val[:, 0]
        vel_v_pred = vel_v_val + output_val[:, 1]
        vel_r_pred = vel_r_val + output_val[:, 2]

        # Almacenar las predicciones
        vel_u_pred_all.extend(vel_u_pred.numpy().flatten())
        vel_v_pred_all.extend(vel_v_pred.numpy().flatten())
        vel_r_pred_all.extend(vel_r_pred.numpy().flatten())

# Convertir las listas de predicciones a arreglos de NumPy
vel_u_pred_all = np.array(vel_u_pred_all)
vel_v_pred_all = np.array(vel_v_pred_all)
vel_r_pred_all = np.array(vel_r_pred_all)

# Graficar los resultados
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t_val_np, vel_u_val_np, label='Velocidad de Surgencia Real')
plt.plot(t_val_np, vel_u_pred_all, label='Velocidad de Surgencia Predicha')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad de Surgencia (m/s)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_val_np, vel_v_val_np, label='Velocidad de Balanceo Real')
plt.plot(t_val_np, vel_v_pred_all, label='Velocidad de Balanceo Predicha')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad de Balanceo (m/s)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_val_np, vel_r_val_np, label='Velocidad de Guiñada Real')
plt.plot(t_val_np, vel_r_pred_all, label='Velocidad de Guiñada Predicha')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad de Guiñada (rad/s)')
plt.legend()

plt.tight_layout()
plt.show()