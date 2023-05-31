import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")

# Assign color names to the colorbar
color_names = ['Setosa', 'Versicolor', 'Virginica']
plt.colorbar(ticks=[0, 1, 2], label='Colors', format=plt.FuncFormatter(lambda val, loc: color_names[int(val)]))

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()

iris = datasets.load_iris()
X = iris.data[:, 2:]  # we only take the last two features.
y = iris.target
print(y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")

# Assign category names to the colorbar
plt.colorbar(ticks=[0, 1, 2], label='Colors', format=plt.FuncFormatter(lambda val, loc: color_names[int(val)]))

plt.xlabel("Petal length")
plt.ylabel("Petal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()

#Using SVC for petals

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el clasificador SVM lineal
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = clf.predict(X_test)

# Imprimir exactitud del clasificador en los datos de prueba
accuracy = clf.score(X_test, y_test)
print("Exactitud en los datos de prueba:", accuracy)

# Datos nuevos para probar el clasificador
X_new = [[5.0, 3.0], [6.0, 2.5], [4.5, 4.0], [5.5, 3.5], [6.5, 3.0],
         [4.8, 3.2], [5.7, 2.8], [6.2, 2.9], [4.9, 3.1], [6.1, 3.3],
         [5.4, 3.7], [6.3, 3.1], [4.7, 3.0], [5.9, 3.2], [6.0, 3.0],
         [4.8, 3.0], [5.6, 2.9], [6.2, 2.8], [4.9, 3.1], [5.8, 3.1]]

# Realizar predicciones en los datos nuevos
y_new_pred = clf.predict(X_new)

# Imprimir las predicciones en los datos nuevos
print("Predicciones en los datos nuevos:")
for i, prediction in enumerate(y_new_pred):
    print("Dato {}: Predicción {}".format(i+1, prediction))
