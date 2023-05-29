from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Cargar un conjunto de datos de ejemplo
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Obtener la cantidad de observaciones y variables
num_observaciones = X.shape[0]
num_variables = X.shape[1]

print("Cantidad de observaciones:", num_observaciones)
print("Cantidad de variables:", num_variables)
print("")

###################################################################################
# Obtener las variables predictorias (características)
features = wine.feature_names

# Imprimir las variables predictorias
print("Variables predictorias:")
for feature in features:
    print(feature)
print("")

###################################################################################
# Crear un clasificador K-NN
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Realizar la validación cruzada con 5 folds y calcular la exactitud
scores = cross_val_score(knn_classifier, X, y, cv=5, scoring='accuracy')

# Imprimir los resultados del modelo k-NN de cada fold 
print("Modelo k-NN")
print("Resultados de cada fold:", scores)

# Calcular la exactitud media del modelo k-NN
mean_accuracy = scores.mean()
print("Exactitud media:", mean_accuracy)
print("")

################################################################################

# Crear un clasificador SVM con kernel lineal
svm_classifierLinear = svm.SVC(kernel='linear', C=1)

# Realizar la validación cruzada con 5 folds y calcular la exactitud
scoresLineal = cross_val_score(svm_classifierLinear, X, y, cv=5, scoring='accuracy')

# Imprimir los resultados del modelo linearl de cada fold 
print("Modelo SVM Lineal")
print("Resultados de cada fold:", scoresLineal)

# Calcular la exactitud media del modelo linearl
mean_accuracyLineal = scoresLineal.mean()
print("Exactitud media:", mean_accuracyLineal)
print("")

################################################################################

# Crear un clasificador SVM con kernel RBF
svm_classifierRBF = svm.SVC(kernel='rbf', C=1)

# Realizar la validación cruzada con 5 folds y calcular la exactitud
scoresRBF= cross_val_score(svm_classifierRBF, X, y, cv=5, scoring='accuracy')

# Imprimir los resultados del modelo SVM RBF de cada fold 
print("Modelo SVM RBF")
print("Resultados de cada fold:", scoresRBF)
    
# Calcular la exactitud media del modelo SVM RBF
mean_accuracyRBF = scoresRBF.mean()
print("Exactitud media:", mean_accuracyRBF)
print("")

################################################################################

# Crear un clasificador de Árbol de Decisión
dt_classifier = DecisionTreeClassifier()

# Realizar la validación cruzada con 5 folds y calcular la exactitud
scoresDT= cross_val_score(dt_classifier, X, y, cv=5, scoring='accuracy')

# Imprimir los resultados del modelo de Árbol de Decisión de cada fold 
print("Modelo árbol de desición")
print("Resultados de cada fold:", scoresDT)
    
# Calcular la exactitud media del modelo de Árbol de Decisión
mean_accuracyDT = scoresDT.mean()
print("Exactitud media:", mean_accuracyDT)
print("")

################################################################################

# Crear un clasificador de Bosques Aleatorios en k = 100
rf_classifier = RandomForestClassifier(n_estimators=100)

scoresRF= cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')

# Imprimir los resultados del modelo de Bosques Aleatorios de cada fold 
print("Modelo de Bosques Aleatorios")
print("Resultados de cada fold:", scoresRF)
    
# Calcular la exactitud media del modelo de Bosques Aleatorios
mean_accuracyRF = scoresRF.mean()
print("Exactitud media:", mean_accuracyRF)
print("")
