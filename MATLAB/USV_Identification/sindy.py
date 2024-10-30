import numpy as np
import scipy.io
from scipy.signal import lfilter, medfilt
import pysindy as ps

# Cargar los datos
data = scipy.io.loadmat('/home/javipc/catkin_ws/src/usv_sim/MATLAB/USV_Identification/muestreo_externo.mat')
t_u = data['T_u'].flatten()  # Fuerza en u
t_v = data['T_v'].flatten()  # Fuerza en v
t_r = data['T_r'].flatten()  # Fuerza en r
v_x = data['vel_u'].flatten()  # Velocidad en u
v_y = data['vel_v'].flatten()  # Velocidad en v
vel_r = data['vel_r'].flatten()  # Velocidad en r

# Parámetros
dt = 0.02  # Tiempo de muestreo
lam = 1  # Parámetro del filtro

# Derivar velocidades para obtener aceleraciones
a_x = np.gradient(v_x, dt)
a_y = np.gradient(v_y, dt)
a_r = np.gradient(vel_r, dt)

# Aplicar filtro de primer orden
b, a = [1], [1, lam * dt - 1]
a_x_filtered = lfilter(b, a, a_x)
a_y_filtered = lfilter(b, a, a_y)
a_r_filtered = lfilter(b, a, a_r)

# Preparar datos de entrada y salida para SINDy
X = np.vstack([t_u, t_v, t_r]).T  # Fuerzas como entradas
Y = np.vstack([a_x_filtered, a_y_filtered, a_r_filtered]).T  # Aceleraciones filtradas como salidas

# Configurar bibliotecas de características
polynomial_library = ps.PolynomialLibrary(degree=2, include_interaction=True)  # Términos constantes, lineales, cuadráticos y cruzados
trig_library = ps.FourierLibrary(n_frequencies=2)  # Términos trigonométricos (sin/cos)

# Crear biblioteca mixta
combined_library = polynomial_library + trig_library

# Configurar y entrenar el modelo SINDy
optimizer = ps.SR3()  # Optimizador SR3 para regularización
model = ps.SINDy(feature_library=combined_library, optimizer=optimizer)
model.fit(X, t=dt, x_dot=Y)
model.print()
