import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn import datasets
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

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

# Conjunto de entrenamiento y prueba
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

# Calcular la matriz de confusión en los datos de prueba
cm = confusion_matrix(y_test, y_pred)

# Función para graficar la matriz de confusión
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusión', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta verdadera')
    plt.xlabel('Etiqueta predicha')

# Definir las etiquetas de las clases
class_labels = ['Setosa', 'Versicolor', 'Virginica']

# Graficar la matriz de confusión
plot_confusion_matrix(cm, classes=class_labels, normalize=True)
plt.show()

# Realizar k-fold cross validation (k = 10)
k = 10
cv = KFold(n_splits=k, shuffle=True, random_state=42)

# Realizar k-fold cross validation y obtener las métricas de evaluación
accuracies = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
precisions = cross_val_score(clf, X, y, cv=cv, scoring='precision_macro')
recalls = cross_val_score(clf, X, y, cv=cv, scoring='recall_macro')

# Calcular la media y desviación estándar de las métricas de evaluación
mean_accuracy = accuracies.mean()
std_accuracy = accuracies.std()
mean_precision = precisions.mean()
std_precision = precisions.std()
mean_recall = recalls.mean()
std_recall = recalls.std()

# Imprimir los resultados
print("Exactitud promedio: {:.3f} (+/- {:.3f})".format(mean_accuracy, std_accuracy))
print("Precisión promedio: {:.3f} (+/- {:.3f})".format(mean_precision, std_precision))
print("Recall promedio: {:.3f} (+/- {:.3f})".format(mean_recall, std_recall))

# Calcular y mostrar la matriz de confusión promedio
confusion_matrices = []
for train_index, test_index in cv.split(X):
    clf.fit(X[train_index], y[train_index])
    y_pred = clf.predict(X[test_index])
    cm = confusion_matrix(y[test_index], y_pred)
    confusion_matrices.append(cm)

average_cm = np.mean(confusion_matrices, axis=0)
class_labels = ['Setosa', 'Versicolor', 'Virginica']
plot_confusion_matrix(average_cm, classes=class_labels, normalize=True)
plt.show()