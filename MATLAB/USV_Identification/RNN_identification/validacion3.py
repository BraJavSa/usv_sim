import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim

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
        vel = np.array([u[k], v[k], r[k]])
        C = np.array([[0, -delta[4] * r[k], -delta[5] * v[k] - delta[2] * r[k]],
                      [delta[4] * r[k], 0, delta[6] * u[k]],
                      [delta[5] * v[k] + delta[2] * r[k], -delta[6] * u[k], 0]])
        d_vel = IM @ (np.array([T_u[k], T_v[k], T_r[k]]) - C @ vel - D @ vel)
        u[k + 1] = u[k] + d_vel[0] * ts
        v[k + 1] = v[k] + d_vel[1] * ts
        r[k + 1] = r[k] + d_vel[2] * ts

    return u, v, r

# Cargar los datos y los valores de delta
data_val = load_data('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/muestreo_externo_2.mat')
delta_values = loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/delta_valores4_1.mat')['delta'].flatten()
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

# Plot the results for velocities
plt.figure(figsize=(12, 16))

# Subplot for surge velocity u
plt.subplot(4, 1, 1)
plt.plot(range(len(vel_u_val_all)), vel_u_val_all, label='Velocidad de avance Real')
plt.plot(range(len(vel_u_pred_rnn)), vel_u_pred_rnn, label='Velocidad de avance Predicha (RNN)')
plt.plot(range(len(vel_u_pred_mlp)), vel_u_pred_mlp, label='Velocidad de avance Predicha (MLP)')
plt.plot(range(len(t)), u_lsq, label='Velocidad de avance Predicha (LSQ)')
plt.xlabel('Índice de Tiempo')
plt.ylabel('Velocidad de Surgencia (m/s)')
plt.legend()

# Subplot for sway velocity v
plt.subplot(4, 1, 2)
plt.plot(range(len(vel_v_val_all)), vel_v_val_all, label='Velocidad transversal Real')
plt.plot(range(len(vel_v_pred_rnn)), vel_v_pred_rnn, label='Velocidad transversal Predicha (RNN)')
plt.plot(range(len(vel_v_pred_mlp)), vel_v_pred_mlp, label='Velocidad transversal Predicha (MLP)')
plt.plot(range(len(t)), v_lsq, label='Velocidad transversal Predicha (LSQ)')
plt.xlabel('Índice de Tiempo')
plt.ylabel('Velocidad de Balanceo (m/s)')
plt.legend()

# Subplot for yaw velocity r
plt.subplot(4, 1, 3)
plt.plot(range(len(vel_r_val_all)), vel_r_val_all, label='Velocidad de Guiñada Real')
plt.plot(range(len(vel_r_pred_rnn)), vel_r_pred_rnn, label='Velocidad de Guiñada Predicha (RNN)')
plt.plot(range(len(vel_r_pred_mlp)), vel_r_pred_mlp, label='Velocidad de Guiñada Predicha (MLP)')
plt.plot(range(len(t)), r_lsq, label='Velocidad de Guiñada Predicha (LSQ)')
plt.xlabel('Índice de Tiempo')
plt.ylabel('Velocidad de Guiñada (rad/s)')
plt.legend()

# Subplot for overall comparison
plt.subplot(4, 1, 4)
plt.plot(range(len(t)), u_lsq, label='Velocidad de avance Predicha (LSQ)')
plt.plot(range(len(t)), v_lsq, label='Velocidad transversal Predicha (LSQ)')
plt.plot(range(len(t)), r_lsq, label='Velocidad de Guiñada Predicha (LSQ)')
plt.xlabel('Índice de Tiempo')
plt.ylabel('Velocidad (m/s o rad/s)')
plt.legend()

plt.tight_layout()
plt.show()
