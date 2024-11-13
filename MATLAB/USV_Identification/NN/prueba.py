import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt

# Cargar los datos de entrenamiento
data = scipy.io.loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/NN/ident_usv.mat')
T_u = data['T_u']  # Fuerza de dirección longitudinal
T_r = data['T_r']  # Fuerza de dirección de rotación
vel_u = data['vel_u']  # Velocidad en dirección longitudinal
vel_v = data['vel_v']  # Velocidad en dirección transversal
vel_r = data['vel_r']  # Velocidad de rotación

# Convertir los datos a tensores
T_u_tensor = torch.tensor(T_u, dtype=torch.float32)
T_r_tensor = torch.tensor(T_r, dtype=torch.float32)
vel_u_tensor = torch.tensor(vel_u, dtype=torch.float32)
vel_v_tensor = torch.tensor(vel_v, dtype=torch.float32)
vel_r_tensor = torch.tensor(vel_r, dtype=torch.float32)

# Desplazar las velocidades (salidas)
vel_u_shifted = torch.roll(vel_u_tensor, shifts=1, dims=0)
vel_u_shifted[0] = 0
vel_v_shifted = torch.roll(vel_v_tensor, shifts=1, dims=0)
vel_v_shifted[0] = 0
vel_r_shifted = torch.roll(vel_r_tensor, shifts=1, dims=0)
vel_r_shifted[0] = 0

# Concatenar las salidas en un solo tensor de tamaño [11500, 3]
target = torch.stack([vel_u_shifted, vel_v_shifted, vel_r_shifted], dim=1)

# Asegurarse de que las dimensiones de target sean [11500, 3] (sin dimensión extra)
target = target.squeeze(-1)  # Elimina la dimensión extra si existe

# Modelo de Red Neuronal
class USVModel(nn.Module):
    def __init__(self):
        super(USVModel, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # 2 entradas: T_u y T_r
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)   # 3 salidas: vel_u_shifted, vel_v_shifted, vel_r_shifted

    def forward(self, T_u, T_r):
        # Concatenar las entradas
        x = torch.cat([T_u, T_r], dim=1)  # [11500, 2]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # [11500, 3]

# Inicializar el modelo, la función de pérdida y el optimizador
model = USVModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Entrenamiento del modelo
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Calcular las predicciones
    pred = model(T_u_tensor.view(-1, 1), T_r_tensor.view(-1, 1))
    
    # Calcular la pérdida (error entre las predicciones y las velocidades desplazadas)
    loss = criterion(pred, target)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluar el modelo con datos nuevos (validación)
# Cargar el nuevo archivo de validación
validation_data = scipy.io.loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/NN/muestreo_externo.mat')

# Extraer las mismas variables del archivo de validación
T_u_val = validation_data['T_u']  # Fuerza de dirección longitudinal
T_r_val = validation_data['T_r']  # Fuerza de dirección de rotación
vel_u_val = validation_data['vel_u']  # Velocidad en dirección longitudinal
vel_v_val = validation_data['vel_v']  # Velocidad en dirección transversal
vel_r_val = validation_data['vel_r']  # Velocidad de rotación

# Convertir los datos de validación a tensores
T_u_val_tensor = torch.tensor(T_u_val, dtype=torch.float32)
T_r_val_tensor = torch.tensor(T_r_val, dtype=torch.float32)
vel_u_val_tensor = torch.tensor(vel_u_val, dtype=torch.float32)
vel_v_val_tensor = torch.tensor(vel_v_val, dtype=torch.float32)
vel_r_val_tensor = torch.tensor(vel_r_val, dtype=torch.float32)

# Desplazar las velocidades (salidas)
vel_u_val_shifted = torch.roll(vel_u_val_tensor, shifts=1, dims=0)
vel_u_val_shifted[0] = 0
vel_v_val_shifted = torch.roll(vel_v_val_tensor, shifts=1, dims=0)
vel_v_val_shifted[0] = 0
vel_r_val_shifted = torch.roll(vel_r_val_tensor, shifts=1, dims=0)
vel_r_val_shifted[0] = 0

# Concatenar las salidas en un solo tensor de tamaño [11500, 3]
target_val = torch.stack([vel_u_val_shifted, vel_v_val_shifted, vel_r_val_shifted], dim=1)

# Asegurarse de que las dimensiones de target sean [11500, 3] (sin dimensión extra)
target_val = target_val.squeeze(-1)  # Elimina la dimensión extra si existe

# Hacer predicciones para los datos de validación
with torch.no_grad():
    model.eval()
    pred_val = model(T_u_val_tensor.view(-1, 1), T_r_val_tensor.view(-1, 1))

# Graficar las velocidades reales y las predicciones
plt.figure(figsize=(12, 8))

# Graficar la velocidad longitudinal (vel_u)
plt.subplot(3, 1, 1)
plt.plot(vel_u_val_shifted.numpy(), label='Velocidad Real (u)')
plt.plot(pred_val[:, 0].numpy(), label='Predicción (u)', linestyle='--')
plt.title('Comparación de Velocidad Longitudinal')
plt.xlabel('Muestras')
plt.ylabel('Velocidad [m/s]')
plt.legend()

# Graficar la velocidad transversal (vel_v)
plt.subplot(3, 1, 2)
plt.plot(vel_v_val_shifted.numpy(), label='Velocidad Real (v)')
plt.plot(pred_val[:, 1].numpy(), label='Predicción (v)', linestyle='--')
plt.title('Comparación de Velocidad Transversal')
plt.xlabel('Muestras')
plt.ylabel('Velocidad [m/s]')
plt.legend()

# Graficar la velocidad de rotación (vel_r)
plt.subplot(3, 1, 3)
plt.plot(vel_r_val_shifted.numpy(), label='Velocidad Real (r)')
plt.plot(pred_val[:, 2].numpy(), label='Predicción (r)', linestyle='--')
plt.title('Comparación de Velocidad de Rotación')
plt.xlabel('Muestras')
plt.ylabel('Velocidad [rad/s]')
plt.legend()

# Mostrar las gráficas
plt.tight_layout()
plt.show()
