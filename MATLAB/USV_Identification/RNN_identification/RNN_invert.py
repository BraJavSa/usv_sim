import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
# Cargar datos
data = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/muestreo_externo_2.mat')
data_val = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/ident_usv_3.mat')

# Extraer variables
vel_u = data['vel_u'].flatten()
vel_v = data['vel_v'].flatten()
vel_r = data['vel_r'].flatten()
vel_u = np.roll(vel_u, -1)
vel_v = np.roll(vel_v, -1)
vel_r = np.roll(vel_r, -1)
T_u = data['T_u'].flatten()
T_r = data['T_r'].flatten()

# Calcular aceleraciones
dt = 0.02
acc_u = np.gradient(vel_u, dt)
acc_v = np.gradient(vel_v, dt)
acc_r = np.gradient(vel_r, dt)

vel_u = vel_u[:-1]
vel_v = vel_v[:-1]
vel_r = vel_r[:-1]
T_u = T_u[:-1]                   
T_r = T_r[:-1]


acc_u = acc_u[:-1]
acc_v = acc_v[:-1]
acc_r = acc_r[:-1]
# Preparar datos de entrada y salida
X = np.vstack([vel_u, vel_v, vel_r, acc_u, acc_v, acc_r]).T
y = np.vstack([T_u, T_r]).T

# Escalar datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.001),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(2)  # Dos salidas: T_u y T_r
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])



# Entrenar modelo
history = model.fit(X_train, y_train, epochs=5000, batch_size=128, verbose=1)

# Validar con data_val
vel_u_val = data_val['vel_u'].flatten()
vel_v_val = data_val['vel_v'].flatten()
vel_r_val = data_val['vel_r'].flatten()
T_u_val = data_val['T_u'].flatten()
T_r_val = data_val['T_r'].flatten()

acc_u_val = np.gradient(vel_u_val, dt)
acc_v_val = np.gradient(vel_v_val, dt)
acc_r_val = np.gradient(vel_r_val, dt)

X_val = np.vstack([vel_u_val, vel_v_val, vel_r_val, acc_u_val, acc_v_val, acc_r_val]).T
X_val_scaled = scaler_X.transform(X_val)

# Predicciones
y_pred_scaled = model.predict(X_val_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Graficar resultados
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(T_u_val, label='T_u Real')
plt.plot(y_pred[:, 0], label='T_u Predicho', linestyle='--')
plt.legend()
plt.title('Comparación T_u')

plt.subplot(2, 1, 2)
plt.plot(T_r_val, label='T_r Real')
plt.plot(y_pred[:, 1], label='T_r Predicho', linestyle='--')
plt.legend()
plt.title('Comparación T_r')

plt.tight_layout()
plt.show()

# Guardar el modelo completo
model.save('/home/javipc/Desktop/mi_modelo.h5')
print("Modelo guardado en '~/Desktop/mi_modelo.h5'")

# Guardar pesos y arquitectura por separado
model.save_weights('/home/javipc/Desktop/pesos_modelo.h5')
print("Pesos del modelo guardados en '~/Desktop/pesos_modelo.h5'")

with open('/home/javipc/Desktop/arquitectura_modelo.json', 'w') as json_file:
    json_file.write(model.to_json())
print("Arquitectura del modelo guardada en '~/Desktop/arquitectura_modelo.json'")




# Guardar escaladores en el escritorio
joblib.dump(scaler_X, '/home/javipc/Desktop/scaler_X.gz')
joblib.dump(scaler_y, '/home/javipc/Desktop/scaler_y.gz')
print("Escaladores guardados en '~/Desktop/'")
