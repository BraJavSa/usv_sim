import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

# Cargar el modelo guardado
model = tf.keras.models.load_model('/home/javipc/Desktop/mi_modelo.h5')

# Cargar los escaladores guardados
scaler_X = joblib.load('/home/javipc/Desktop/scaler_X.gz')
scaler_y = joblib.load('/home/javipc/Desktop/scaler_y.gz')

# Cargar nuevos datos
data_nuevos = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/sim_val.mat')

# Extraer variables de los nuevos datos
vel_u_nuevos = data_nuevos['vel_u'].flatten()
vel_v_nuevos = data_nuevos['vel_v'].flatten()
vel_r_nuevos = data_nuevos['vel_r'].flatten()
T_u_nuevos = data_nuevos['T_u'].flatten()
T_r_nuevos = data_nuevos['T_r'].flatten()

# Calcular aceleraciones de los nuevos datos
dt = 0.02
acc_u_nuevos = np.gradient(vel_u_nuevos, dt)
acc_v_nuevos = np.gradient(vel_v_nuevos, dt)
acc_r_nuevos = np.gradient(vel_r_nuevos, dt)

# Preparar datos de entrada
X_nuevos = np.vstack([vel_u_nuevos, vel_v_nuevos, vel_r_nuevos, acc_u_nuevos, acc_v_nuevos, acc_r_nuevos]).T

# Escalar datos usando los escaladores cargados
X_nuevos_scaled = scaler_X.transform(X_nuevos)

# Hacer predicciones
y_pred_nuevos_scaled = model.predict(X_nuevos_scaled)
y_pred_nuevos = scaler_y.inverse_transform(y_pred_nuevos_scaled)

# Graficar resultados con los nuevos datos
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(T_u_nuevos, label='T_u Real')
plt.plot(y_pred_nuevos[:, 0], label='T_u Predicho', linestyle='--')
plt.legend()
plt.title('Comparación T_u (Nuevos Datos)')

plt.subplot(2, 1, 2)
plt.plot(T_r_nuevos, label='T_r Real')
plt.plot(y_pred_nuevos[:, 1], label='T_r Predicho', linestyle='--')
plt.legend()
plt.title('Comparación T_r (Nuevos Datos)')

plt.tight_layout()
plt.show()
