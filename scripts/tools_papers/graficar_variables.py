import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

# Ruta del archivo .mat
file_path = '/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/RNN_identification/ident_usv_3.mat'

# Cargar datos del archivo
data = loadmat(file_path)

# Extraer variables
T_u = data['T_u'].flatten()  # Convertir a un array unidimensional
T_r = data['T_r'].flatten()  # Convertir a un array unidimensional
ts = data['ts'].item()  # Obtener ts como un escalar

# Crear vector de tiempo ajustado para 0 a 90 segundos
time = np.arange(0, len(T_u) * ts, ts)

# Configurar figura con dos subgráficos
plt.figure(figsize=(10, 8))

# Gráfico de T_u (Thrust Force)
plt.subplot(2, 1, 1)  # 2 filas, 1 columna, posición 1
plt.plot(time, T_u, label='T_u')
plt.xlabel('Time [s]')
plt.ylabel('Thrust Force [N]')
plt.title('Thrust Force vs Time')
plt.xlim(0, len(T_u) * ts)  # Limitar el eje x a 0-90 segundos
plt.grid(True)
plt.legend(fontsize=14)  # Aumentar tamaño de fuente de la leyenda

# Gráfico de T_r (Yaw Moment)
plt.subplot(2, 1, 2)  # 2 filas, 1 columna, posición 2
plt.plot(time, T_r, color='tab:orange', label='T_r')
plt.xlabel('Time [s]')
plt.ylabel('Yaw Moment [Nm]')
plt.title('Yaw Moment vs Time')
plt.xlim(0, len(T_u) * ts)  # Limitar el eje x a 0-90 segundos
plt.grid(True)
plt.legend(fontsize=14)  # Aumentar tamaño de fuente de la leyenda

# Ajustar diseño
plt.tight_layout()
plt.show()
