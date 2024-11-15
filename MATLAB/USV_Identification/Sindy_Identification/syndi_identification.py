import numpy as np
import warnings
from copy import copy
import pysindy as ps
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import LinAlgWarning

def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    warnings.filters = filters


# Cargar el archivo .mat
data = loadmat('muestreo_externo.mat')
t = data['t'].flatten()  # Vector de tiempo
ts = data['ts']# Tiempo de muestreo
vel_u = data['vel_u'].flatten()# Velocidad en u
vel_v = data['vel_v'].flatten()# Velocidad en v
vel_r = data['vel_r'].flatten()# Velocidad en r
T_u = data['T_u'].flatten()# Torque en u
T_v = data['T_v'].flatten()# Torque en u
T_r = data['T_r'].flatten() # Torque en r

# Definir la duración y el rango de datos para el entrenamiento
ext = 1500
init = 10

# Rango de tiempo
t_vector = t[init:ext]
dt = ts

# Velocidades
velocities = np.vstack((vel_u[init:ext], vel_v[init:ext], vel_r[init:ext]))

# Torques aplicados
torques = np.vstack((T_u[init:ext],T_v[init:ext], T_r[init:ext]))


u = torques


v1_p = velocities
v1_p = v1_p.T



u_train = u[:,0:ext]

u_train = u_train.T





u_train_multi = []



# Agrega los vectores a la lista x_train_multi
u_train_multi.append(u_train)
u_train_multi.append(u_train)
u_train_multi.append(u_train)
u_train_multi.append(u_train)


library1 = ps.PolynomialLibrary(degree=1)
library2 = ps.FourierLibrary(n_frequencies=2)
library3 = ps.IdentityLibrary(  )
lib_generalized = ps.GeneralizedLibrary([library1, library2])
lib_generalized.fit(v1_p)

optimizador = ps.SR3(trimming_fraction=0.1)


variables = ['u', 'v', 'r']

# Define las funciones de la biblioteca
def identity(x):
    return x

def seno(x):
    if 'u' in x.tolist():
        return str(x)
    else:
        return np.sin(x)
    
def coseno(x):
    if 'u' in x.tolist():
        return str(x)
    else:
        return np.cos(x)



# Define los nombres de las funciones
def identity_name(x):
    return str(x)

def seno_name(x):
    if 'u' not in x:
        return "sin("+str(x)+")"
    else:
        return str(x)
    
def coseno_name(x):
    if 'u' not in x:
        return "cos("+str(x)+")"
    else:
        return str(x)

library_functions = [
    identity,
    seno,
    coseno,
]

library_function_names = [
    identity_name,
    seno_name,
    coseno_name
]

custom_library = ps.CustomLibrary(
    library_functions=library_functions, 
    function_names=library_function_names
)

model = ps.SINDy(
    
    optimizer=ps.STLSQ(threshold=0.1, alpha=.05, verbose=True),
    feature_library=custom_library,
    differentiation_method = ps.SINDyDerivative(kind="kalman", alpha=0.05),
    feature_names=["u", "v","r" ], 
)




model.fit(v1_p,  t=t_vector, u = u_train, multiple_trajectories=False)
model.print()
feature_names = model.get_feature_names()


for name in feature_names:
    print(name)



x0_test = v1_p[0,:]



x_test_sim = model.simulate(x0=x0_test, t=t_vector, u=u_train )

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(v1_p, u=u_train)  

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(v1_p, t=dt)

"""
# Plot original data and model prediction
plt.figure(figsize=(10, 6))


# Plot original data
plt.subplot(3, 2, 1)
plt.plot(range(len(x_dot_test_computed[:, 0])), x_dot_test_computed[:, 0], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 0])), x_dot_test_predicted [:, 0], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 0]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot second variable
plt.subplot(3, 2, 2)
plt.plot(range(len(x_dot_test_computed[:, 1])), x_dot_test_computed[:, 1], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 1])), x_dot_test_predicted [:, 1], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 1]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot third variable
plt.subplot(3, 2, 3)
plt.plot(range(len(x_dot_test_computed[:, 2])), x_dot_test_computed[:, 2], label='Original Data', color='blue')
plt.plot(range(len(x_dot_test_predicted [:, 2])), x_dot_test_predicted [:, 2], label='Model Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 2]) and Model Prediction')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





# Plot original data and model prediction
plt.figure(figsize=(10, 6))


# Plot original data
plt.subplot(3, 2, 1)
plt.plot(range(len(x_test_sim[:, 0])), x_test_sim[:, 0], label='Prediction Data', color='blue')
plt.plot(range(len(v1_p[:, 0])), v1_p[:, 0], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 0]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot second variable
plt.subplot(3, 2, 2)
plt.plot(range(len(x_test_sim[:, 1])), x_test_sim[:, 1], label='Prediction Data', color='blue')
plt.plot(range(len(v1_p[:, 1])), v1_p[:, 1], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 1]) and Model Prediction')
plt.legend()
plt.grid(True)

# Plot third variable
plt.subplot(3, 2, 3)
plt.plot(range(len(x_test_sim[:, 2])), x_test_sim[:, 2], label='Prediction Data', color='blue')
plt.plot(range(len(v1_p[:, 2])), v1_p[:, 2], label='Model Real', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Variable')
plt.title('Original Data ([:, 2]) and Model Prediction')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
"""
