
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim

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

# Crear el modelo de red neuronal
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

# Cargar y preparar datos
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

# Cargar datos
data = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/ident_usv_2.mat')
data_val = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/muestreo_externo_2.mat')

train_dataset, T_u, T_r, pass_vel_u, pass_vel_v, pass_vel_r, pass2_vel_u, pass2_vel_v, pass2_vel_r, vel_u, vel_v, vel_r = create_dataset(data)
val_dataset, T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val, vel_u_val, vel_v_val, vel_r_val = create_dataset(data_val)

# Hiperparámetros
input_size = 8
hidden_size = 200
output_size = 3
learning_rate = 0.001
num_epochs = 80

# Crear el modelo, función de pérdida y optimizador
model = USVNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Crear el data loader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Bucle de entrenamiento
for epoch in range(num_epochs):
    for i, (T_u, T_r, pass_vel_u, pass_vel_v, pass_vel_r, pass2_vel_u, pass2_vel_v, pass2_vel_r, vel_u, vel_v, vel_r) in enumerate(train_dataloader):
        # Preparar tensores de entrada y objetivo
        input_tensor = torch.stack([T_u, T_r, pass_vel_u, pass_vel_v, pass_vel_r, pass2_vel_u, pass2_vel_v, pass2_vel_r], dim=1)
        target_tensor = torch.stack([vel_u, vel_v, vel_r], dim=1)

        # Forward pass
        output = model(input_tensor)

        # Calcular pérdida
        loss = criterion(output, target_tensor)

        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    if loss.item() <= 0.00025:
        break

# Guardar el modelo
model_path = '/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/usv_nn_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en '{model_path}'")

# Evaluar el modelo
model.eval()

# Inicializar listas vacías para predicciones y valores reales
vel_u_pred_all = []
vel_v_pred_all = []
vel_r_pred_all = []
vel_u_val_all = []
vel_v_val_all = []
vel_r_val_all = []

with torch.no_grad():
    for (T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val,
         vel_u_val, vel_v_val, vel_r_val) in val_dataloader:
        # Preparar tensor de entrada
        input_tensor_val = torch.stack([T_u_val, T_r_val, pass_vel_u_val, pass_vel_v_val, pass_vel_r_val, pass2_vel_u_val, pass2_vel_v_val, pass2_vel_r_val], dim=1)

        # Realizar predicción
        output_val = model(input_tensor_val)

        # Extraer predicciones de velocidades
        vel_u_pred = output_val[:, 0].numpy().flatten()
        vel_v_pred = output_val[:, 1].numpy().flatten()
        vel_r_pred = output_val[:, 2].numpy().flatten()

        # Almacenar predicciones y valores reales
        vel_u_pred_all.extend(vel_u_pred)
        vel_v_pred_all.extend(vel_v_pred)
        vel_r_pred_all.extend(vel_r_pred)
        vel_u_val_all.extend(vel_u_val.numpy().flatten())
        vel_v_val_all.extend(vel_v_val.numpy().flatten())
        vel_r_val_all.extend(vel_r_val.numpy().flatten())

# Convertir listas a arrays numpy
vel_u_pred_all = np.array(vel_u_pred_all)
vel_v_pred_all = np.array(vel_v_pred_all)
vel_r_pred_all = np.array(vel_r_pred_all)
vel_u_val_all = np.array(vel_u_val_all)
vel_v_val_all = np.array(vel_v_val_all)
vel_r_val_all = np.array(vel_r_val_all)

# Graficar los resultados
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(range(len(vel_u_val_all)), vel_u_val_all, label='Velocidad de avance Real')
plt.plot(range(len(vel_u_pred_all)), vel_u_pred_all, label='Velocidad de avance Predicha')
plt.xlabel('Índice de Tiempo')
plt.ylabel('Velocidad de Surgencia (m/s)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(len(vel_v_val_all)), vel_v_val_all, label='Velocidad transversal Real')
plt.plot(range(len(vel_v_pred_all)), vel_v_pred_all, label='Velocidad transversal Predicha')
plt.xlabel('Índice de Tiempo')
plt.ylabel('Velocidad de Balanceo (m/s)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(range(len(vel_r_val_all)), vel_r_val_all, label='Velocidad de Guiñada Real')
plt.plot(range(len(vel_r_pred_all)), vel_r_pred_all, label='Velocidad de Guiñada Predicha')
plt.xlabel('Índice de Tiempo')
plt.ylabel('Velocidad de Guiñada (rad/s)')
plt.legend()

plt.tight_layout()
plt.show()
