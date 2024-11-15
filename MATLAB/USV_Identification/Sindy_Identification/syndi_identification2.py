#!/usr/bin/env python3

import numpy as np
import pysindy as ps

# Datos de ejemplo para las características (esto debe ser reemplazado con tus datos reales)
# Asegúrate de que estos datos estén formateados correctamente
# Por ejemplo, u, v, r podrían ser velocidades y tasa de giro de tu vehículo
t_vector = np.linspace(0, 10, 100)  # Tiempo de 0 a 10 segundos, 100 puntos
velocities = np.random.randn(3, 100)  # Generar datos aleatorios para u, v, r
torques = np.random.randn(3, 100)  # Generar torques de ejemplo

# Definir las funciones personalizadas
def identity(x):
    return x

def seno(x):
    return np.sin(x)

def coseno(x):
    return np.cos(x)

# Asegúrate de que el número de funciones coincida con las características
library_functions = [
    identity,  # Función identidad
    seno,      # Función seno
    coseno     # Función coseno
]

library_function_names = [
    lambda x: str(x),
    lambda x: f"sin({str(x)})",
    lambda x: f"cos({str(x)})"
]

# Crear la biblioteca personalizada correctamente alineada
custom_library = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_function_names
)

# Definir las características, en este caso, u, v, r
feature_names = ['u', 'v', 'r']

# Crear el modelo SINDy
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1, alpha=0.05, verbose=True),
    feature_library=custom_library,
    differentiation_method=ps.SINDyDerivative(kind="kalman", alpha=0.05),
    feature_names=feature_names
)

# Calcular la derivada de los datos utilizando diferencia finita
x_dot_precomputed = ps.FiniteDifference()._differentiate(velocities.T, t_vector)

# Entrenamiento del modelo
model.fit(velocities.T, t=t_vector, u=torques.T, multiple_trajectories=False)

# Imprimir las ecuaciones del modelo
model.print()
eature_names = model.get_feature_names()

for name in feature_names:
    print(name)



x0_test = v1_p[0,:]



x_test_sim = model.simulate(x0=x0_test, t=t_vector, u=u_train )

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(v1_p, u=u_train)  

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(v1_p, t=dt)


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