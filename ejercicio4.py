import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

####################################################################################################################################

# Cargar el archivo de texto y procesar los datos
file_path = "c:/Users/Admin/Desktop/Semestre 6/deep/ajustesModeloClasificacion/misterious_data_2.txt"
data = np.loadtxt(file_path)
x = data[:, 1:]
y = data[:, 0]

####################################################################################################################################

# Obtener informacion de los datos misterios
print("-------------------------------------------")
print('Features:', len(x[0]))
print('Classes:', len(set(y)))
print("-------------------------------------------")

####################################################################################################################################
#Parte 1
#Calcular la exactitud de kNN para k = 1 a 40


best_k = None
best_accuracy = 0.0

for k in range(1, 41):

    # Crear un clasificador K-NN
    knn_Classifier = KNeighborsClassifier(n_neighbors=k)

    # Realizar la validación cruzada con 5 folds y calcular la exactitud
    scoresKNN = cross_validate(knn_Classifier, x, y, cv=5, scoring='accuracy')

    # Calcular la exactitud media del modelo k-NN
    average_accuracy = np.mean(scoresKNN['test_score'])

    #Imprimir la exactitud media 
    print("Exactitud media en los datos seleccionados para k =", k, ": ",average_accuracy)

    # Actualizar el valor más alto de la media
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_k = k

print("-------------------------------------------")
print("El valor más alto de media es:", best_accuracy)
print("Se obtuvo con k =", best_k)
print("-------------------------------------------")
print(" ")

####################################################################################################################################

best_c = None
best_average = 0.0

c = 0.000001
while c < 0.0001:
    
    # Crear un clasificador SVM lineal
    svm_classifierLinear = svm.SVC(kernel="linear", C = c) 

    # Realizar la validación cruzada con 5 folds y calcular la exactitud
    scoresLin = cross_validate(svm_classifierLinear, x, y, cv=5, scoring=('accuracy', 'recall_micro'))
    
    # Calcular la exactitud media del modelo SVM lineal
    average = np.mean(scoresLin['test_accuracy'])

    #Imprimir la exactitud media 
    print("Parametro en c =", c,": ", average)

    # Actualizar el valor más alto de la media
    if average > best_average:
        best_average = average
        best_c = c

    c += 0.000001

print(" ")
print("-------------------------------------------")
print("El valor más alto de media es:", best_average)
print("Se obtuvo con C =", best_c)
print("-------------------------------------------")
print(" ") 

####################################################################################################################################

best_cR = None
best_avgRBF = 0.0

g = 0.000001
while g <= 0.0002:

    # Crear un clasificador SVM rbf
    svm_classifierRBF  = svm.SVC(kernel="rbf", gamma= g )

    # Realizar la validación cruzada con 5 folds y calcular la exactitud
    scoresRBF = cross_validate(svm_classifierRBF , x, y, cv=5, scoring=('accuracy', 'recall_micro'))

    # Calcular la exactitud media del modelo SVM lineal
    averageR = np.mean(scoresRBF['test_accuracy'])
    
    #Imprimir la exactitud media 
    print("Parametro en c =", g,": ", averageR)

    # Actualizar el valor más alto de la media
    if averageR > best_avgRBF:
        best_avgRBF = averageR
        best_cR = g

    g += 0.000001

print(" ")
print("-------------------------------------------")
print("El valor más alto de media es:", best_cR)
print("Se obtuvo con C =", best_avgRBF)
print("-------------------------------------------")
print(" ")
    


