from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split

# Cargar el archivo de texto y procesar los datos
file_path = "c:/Users/Admin/Desktop/Semestre 6/deep/ajustesModeloClasificacion/misterious_data_1.txt"
data = np.genfromtxt(file_path, delimiter='\t')
X = data[:, :-1]
y = data[:, -1]

# Obtener el número de observaciones
num_observaciones = X.shape[0]

#Porcentaje de la base de datos que se utilizara
porc = .30
obs = int(num_observaciones * porc)

# Division de los datos en el porcentaje a usar
score = np.random.choice(range(num_observaciones), size= obs, replace=False)

# Seleccionar aleatoriamente elos datos dependiendo el porcentaje usado
X_2 = X[score]
y_2 = y[score]

# Crear un regresor SVM lineal
svm_classifierLinear = SVR(kernel='linear', C=1)

# Realizar la validación cruzada con 5 folds en los datos seleccionados
scores = cross_val_score(svm_classifierLinear, X_2, y_2, cv=40)

# Imprimir los resultados de cada fold en los datos seleccionados
print("Resultados de cada fold en los datos seleccionados:", scores)

# Calcular la puntuación media en los datos seleccionados
mean_score = scores.mean()
print("Puntuación media en los datos seleccionados con",porc,"de la base de datos:", mean_score)
