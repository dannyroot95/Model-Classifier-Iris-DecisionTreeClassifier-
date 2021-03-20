
# Modelo clasificador de tipos de Flor de Iris : Setosa, Versicolor, Virginica utilizando arboles de descición.

# Importamos las librerías

import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Cargamos el dataset

iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Separar del dataframe dos atributos y las etiquetas. Llamar X a los features e y a las etiquetas.

X = data[['petal length (cm)', 'petal width (cm)']]
# X = data.drop("target", axis=1)
y = data.target

# Importamos y creamos un un modelo de clasificación de vecinos más cercanos con los argumentos por defecto. 

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

# Entrenar el clasificador que creaste.

clf.fit(X,y)

# Predecir con el modelo las etiquetas sobre todo X.

y_pred = clf.predict(X)

# Evaluar la performance del modelo usando accuracy_score y confusion_matrix

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y,y_pred))

confusion_matrix(y, y_pred)

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X, y,cmap=plt.cm.Blues,normalize=None)

# Visualizar las fronteras de decisión obtenidas

plt.figure()
ax = sns.scatterplot(X.iloc[:,0], X.iloc[:,1], hue=y.values, palette='Set2')
plt.legend().remove()


xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                      np.linspace(*ylim, num=200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

contours = ax.contourf(xx, yy, Z, alpha=0.3, cmap = 'Set2')
plt.tight_layout()
# plt.savefig('arbol_iris.png', dpi = 400)
plt.show()

# Creamos un objeto arbol
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)

# Entrenamos el modelo
clf.fit(X, y)

# Predecimos sobre nuestro set
for i in range(10):
    clf.fit(X, y)
    y_pred = clf.predict(X)

# Comaparamos con las etiquetas reales
    print('Accuracy:', accuracy_score(y_pred,y))

# Dibujamos el arbol

plt.figure(figsize = (10,8))
tree.plot_tree(clf, filled = True, feature_names= X.columns)
plt.show()

# Mostramos la importancia de cada caracteristica

importances = clf.feature_importances_
columns = X.columns
sns.barplot(columns, importances)
plt.title('Importancia de cada Feature')
plt.grid()
plt.show()

# En conclusion podemos observar la importacia de la altura al clasificar el tipo de flor