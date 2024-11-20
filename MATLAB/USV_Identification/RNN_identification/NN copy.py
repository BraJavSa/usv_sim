import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim

class USVDataset(torch.utils.data.Dataset):
    def __init__(self, vel_u, vel_v, vel_r, acc_u, acc_v, acc_r, T_u, T_r):
        self.vel_u = torch.tensor(vel_u, dtype=torch.float32)
        self.vel_v = torch.tensor(vel_v, dtype=torch.float32)
        self.vel_r = torch.tensor(vel_r, dtype=torch.float32)
        self.acc_u = torch.tensor(acc_u, dtype=torch.float32)
        self.acc_v = torch.tensor(acc_v, dtype=torch.float32)
        self.acc_r = torch.tensor(acc_r, dtype=torch.float32)
        self.T_u = torch.tensor(T_u, dtype=torch.float32)
        self.T_r = torch.tensor(T_r, dtype=torch.float32)

    def __len__(self):
        return len(self.vel_u)

    def __getitem__(self, idx):
        return (self.vel_u[idx], self.vel_v[idx], self.vel_r[idx], self.acc_u[idx], self.acc_v[idx], self.acc_r[idx], self.T_u[idx], self.T_r[idx])

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

def create_dataset(data):
    vel_u = data['vel_u'].flatten()
    vel_v = data['vel_v'].flatten()
    vel_r = data['vel_r'].flatten()
    acc_u = data['acc_u'].flatten()
    acc_v = data['acc_v'].flatten()
    acc_r = data['acc_r'].flatten()
    T_u = data['T_u'].flatten()
    T_r = data['T_r'].flatten()

    dataset = USVDataset(vel_u, vel_v, vel_r, acc_u, acc_v, acc_r, T_u, T_r)
    return dataset

# Cargar datos
data = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/ident_usv_3.mat')
data_val = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/muestreo_externo_3.mat')

train_dataset = create_dataset(data)
val_dataset = create_dataset(data_val)

# Hiperparámetros
input_size = 6  # 6 entradas: vel_u, vel_v, vel_r, acc_u, acc_v, acc_r
hidden_size = 200
output_size = 2  # 2 salidas: T_u, T_r
learning_rate = 0.001
num_epochs = 40

# Crear el modelo, función de pérdida y optimizador
model = USVNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Crear el data loader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Bucle de entrenamiento
for epoch in range(num_epochs):
    for i, (vel_u, vel_v, vel_r, acc_u, acc_v, acc_r, T_u, T_r) in enumerate(train_dataloader):
        # Preparar tensores de entrada y objetivo
        input_tensor = torch.stack([vel_u, vel_v, vel_r, acc_u, acc_v, acc_r], dim=1)
        target_tensor = torch.stack([T_u, T_r], dim=1)

        # Forward pass
        output = model(input_tensor)

        # Calcular pérdida
        loss = criterion(output, target_tensor)

        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    if loss.item() <= 0.00007:
        break

# Guardar el modelo
model_path = '/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/usv_inver_nn_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en '{model_path}'")

# Evaluar el modelo
model.eval()

# Inicializar listas vacías para predicciones y valores reales
T_u_pred_all = []
T_r_pred_all = []
T_u_val_all = []
T_r_val_all = []

with torch.no_grad():
    for (vel_u_val, vel_v_val, vel_r_val, acc_u_val, acc_v_val, acc_r_val, T_u_val, T_r_val) in val_dataloader:
        # Preparar tensor de entrada
        input_tensor_val = torch.stack([vel_u_val, vel_v_val, vel_r_val, acc_u_val, acc_v_val, acc_r_val], dim=1)

        # Realizar predicción
        output_val = model(input_tensor_val)

        # Extraer predicciones de T_u y T_r
        T_u_pred = output_val[:, 0].numpy().flatten()
        T_r_pred = output_val[:, 1].numpy().flatten()

        # Almacenar predicciones y valores reales
        T_u_pred_all.extend(T_u_pred)
        T_r_pred_all.extend(T_r_pred)
        T_u_val_all.extend(T_u_val.numpy().flatten())
        T_r_val_all.extend(T_r_val.numpy().flatten())

# Convertir listas a arrays numpy
T_u_pred_all = np.array(T_u_pred_all)
T_r_pred_all = np.array(T_r_pred_all)
T_u_val_all = np.array(T_u_val_all)
T_r_val_all = np.array(T_r_val_all)

# Graficar los resultados
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(range(len(T_u_val_all)), T_u_val_all, label='T_u Real')
plt.plot(range(len(T_u_pred_all)), T_u_pred_all, label='T_u Predicho')
plt.xlabel('Índice de Tiempo')
plt.ylabel('T_u')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(len(T_r_val_all)), T_r_val_all, label='T_r Real')
plt.plot(range(len(T_r_pred_all)), T_r_pred_all, label='T_r Predicho')
plt.xlabel('Índice de Tiempo')
plt.ylabel('T_r')
plt.legend()

plt.tight_layout()
plt.show()
