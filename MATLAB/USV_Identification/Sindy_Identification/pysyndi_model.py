import numpy as np
import warnings
from copy import copy
import pysindy as ps
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error
import math

import matplotlib.pyplot as plt
from scipy.io import loadmat

from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning

def ignore_specific_warnings():
  filters = copy(warnings.filters)
  warnings.filterwarnings("ignore", category=ConvergenceWarning)
  warnings.filterwarnings("ignore", category=LinAlgWarning)
  warnings.filterwarnings("ignore", category=UserWarning)
  yield
  warnings.filters = filters

# Load data from 'muestreo_externo.mat'
data = loadmat('ident_usv.mat')

t = data['t'].flatten()  # Flatten t for easy handling
ts = data['ts'][0, 0]  # Extract sampling time

# Extract velocity and torque data
vel_u = data['vel_u'].flatten()
vel_v = data['vel_v'].flatten()
vel_r = data['vel_r'].flatten()

T_u = data['T_u'].flatten()
T_r = data['T_r'].flatten()

# Specify the desired state variables (velocities)
v1 = np.vstack((vel_u, vel_v, vel_r)).T
library1 = ps.PolynomialLibrary(degree=1)
library2 = ps.FourierLibrary(n_frequencies=2)
library3 = ps.IdentityLibrary()
lib_generalized = ps.GeneralizedLibrary([library1, library2])
lib_generalized.fit(v1)

optimizador = ps.SR3(trimming_fraction=0.1)
state_variables = ["vel_u", "vel_v", "vel_r"]

# Define feature library (consider experimenting with different options)
library = ps.PolynomialLibrary(degree=2)  # Try different polynomial degrees

# Definición de funciones y librerías
def identity(x):
    return x

def seno(x):
    return np.sin(x)

def coseno(x):
    return np.cos(x)

library_functions = [identity, seno, coseno]
library_function_names = ['identity', 'sin', 'cos']
custom_library = ps.CustomLibrary(library_functions, library_function_names)

# Creación del modelo SINDy
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1, alpha=0.05, verbose=True),
    feature_library=custom_library,
    differentiation_method=ps.SINDyDerivative(kind="kalman", alpha=0.05),
    feature_names=state_variables,
)

# Ajuste del modelo
model.fit(np.vstack([vel_u, vel_v, vel_r]).T, t=t, u=np.vstack([T_u, T_r]).T if 'T_u' in data.keys() and 'T_r' in data.keys() else None)

x_dot_predicted = model.predict(np.vstack([vel_u, vel_v, vel_r]).T, u=np.vstack([T_u, T_r]).T if 'T_u' in data.keys() and 'T_r' in data.keys() else None)

# Plot original data and model prediction for velocities
plt.figure(figsize=(10, 6))

# Plot u velocity
plt.subplot(3, 1, 1)
plt.plot(t, vel_u, label='Original u', color='blue')
plt.plot(t, x_dot_predicted[:, 0], label='Predicted u', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('u Velocity')
plt.title('Original u Velocity and Prediction')
plt.legend()
plt.grid(True)

# Plot v velocity
plt.subplot(3, 1, 2)
plt.plot(t, vel_v, label='Original v', color='blue')
plt.plot(t, x_dot_predicted[:, 1], label='Predicted v', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('v Velocity')
plt.title('Original v Velocity and Prediction')
plt.legend()
plt.grid(True)

# Plot r velocity
plt.subplot(3, 1, 3)
plt.plot(t, vel_r, label='Original r', color='blue')
plt.plot(t, x_dot_predicted[:, 2], label='Predicted r', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('r Velocity')
plt.title('Original r Velocity and Prediction')
plt.legend()
plt.grid(True)

plt.show()