import numpy as np
from scipy.io import loadmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Cargar datos desde MATLAB
data_val = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/ident_usv_2.mat')
data_train = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/muestreo_externo_2.mat')
#data_val = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/muestreo_externo_2.mat')

# Extraer variables de entrenamiento y validación
variables = ['vel_u', 'vel_v', 'vel_r', 'acc_u', 'acc_v', 'acc_r', 'T_u', 'T_r', 'pass_vel_u', 'pass_vel_v', 'pass_vel_r', 'pass2_vel_u', 'pass2_vel_v', 'pass2_vel_r']
train_data = {var: data_train[var].flatten() for var in variables}
val_data = {var: data_val[var].flatten() for var in variables}

# Entradas y salidas de entrenamiento
X_train = np.column_stack([train_data[var] for var in variables if var != 'T_u' and var != 'T_r'])  # Incluir todas las variables excepto 'T_u' y 'T_r'
Y_train = np.column_stack([train_data['T_u'], train_data['T_r']])

# Entradas y salidas de validación
X_val = np.column_stack([val_data[var] for var in variables if var != 'T_u' and var != 'T_r'])  # Incluir todas las variables excepto 'T_u' y 'T_r'
Y_val = np.column_stack([val_data['T_u'], val_data['T_r']])

# Definir el modelo
model = Sequential([
    Dense(50, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.1),
    Dense(25, activation='relu'),
    Dropout(0.1),
    Dense(5, activation='relu'),
    Dense(Y_train.shape[1])
])

# Compilar el modelo
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')

# Entrenar el modelo
model.fit(X_train, Y_train, epochs=5, batch_size=20)

# Evaluar el modelo
loss = model.evaluate(X_val, Y_val)
print(f'Pérdida en el conjunto de validación: {loss}')

# Predecir las salidas para el conjunto de validación
Y_pred = model.predict(X_val)

# Graficar torques reales y predichos
time = np.arange(0, len(val_data['T_u'])*0.02, 0.02)
plt.figure(figsize=(12, 8))

# Graficar T_u
plt.subplot(2, 1, 1)
plt.plot(time, val_data['T_u'], label='T_u real')
plt.plot(time, Y_pred[:, 0], label='T_u predicho')
plt.xlabel('Tiempo (s)')
plt.ylabel('T_u')
plt.legend()
plt.title('Torques reales y predichos')

# Graficar T_r
plt.subplot(2, 1, 2)
plt.plot(time, val_data['T_r'], label='T_r real')
plt.plot(time, Y_pred[:, 1], label='T_r predicho')
plt.xlabel('Tiempo (s)')
plt.ylabel('T_r')
plt.legend()

plt.tight_layout()
plt.show()
