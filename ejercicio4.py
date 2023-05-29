from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split

# Cargar el archivo de texto y procesar los datos
file_path = "c:/Users/Admin/Desktop/Semestre 6/deep/ajustesModeloClasificacion/misterious_data_2.txt"
data = np.genfromtxt(file_path, delimiter='\t')
X = data[:, :-1]
y = data[:, -1]

# Crear un regresor k-NN
knn_regressor = KNeighborsRegressor(n_neighbors=5)

# Realizar la validación cruzada con 5 folds en los datos seleccionados
scores = cross_val_score(knn_regressor, X, y, cv=10)

# Imprimir los resultados de cada fold en los datos seleccionados
print("Resultados de cada fold en los datos seleccionados:", scores)

# Calcular la puntuación media en los datos seleccionados
mean_score = scores.mean()
print("Puntuación media en los datos seleccionados:", mean_score)
