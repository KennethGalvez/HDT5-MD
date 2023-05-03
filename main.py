"""
Como asistentes del herbario, podemos decir que este código es funcional y realiza un buen trabajo pero creemos que hay algunas cosas que se pueden mejorar para que sea más eficiente y legible:
El código original tiene una serie de importaciones redundantes que no se utilizan. Decidimos eliminar cualquiera de las importaciones que no se estén utilizando actualmente en el código.
Agrupamos las importaciones y las funciones en diferentes bloques para facilitar la comprensión y que sea todo mas claro.
Mucho del código usa variables que no se entiende en que momento fueron definidas como X_train e Y_train, agrupamos todo de manera en que se entienda en el codigo de donde es que cada cosa aparece.
Nos tomamos la libertad de optimizar el codigo como al momento de cargar los datos o realizar los modelos para mejorar el tiempo a comparacion del original.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Ignorar warnings
warnings.filterwarnings("ignore")

# Cargar data
url = 'https://jupyterlite.anaconda.cloud/b0df9a1c-3954-4c78-96e6-07ab473bea1a/files/iris/iris.csv'
iris_data = pd.read_csv(url)

# Exploracion basica de la data
print(f"Shape of the data: {iris_data.shape}\n")
print(f"First 10 rows of the data:\n{iris_data.head(10)}\n")
print(f"Last 10 rows of the data:\n{iris_data.tail(10)}\n")
print(f"Basic statistics about the data:\n{iris_data.describe()}\n")

# Procesamiento de la data
iris_data = iris_data.drop('Id', axis=1)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print(f"Data after removing 'Id' column and renaming columns:\n{iris_data.head(20)}\n")
print(f"Number of instances for each species:\n{iris_data['species'].value_counts()}\n")

# Vizualizacion de la data
sns.set(style="ticks", color_codes=True)
sns.pairplot(iris_data, hue="species", markers=["o", "s", "D"])
plt.show()

sns.boxplot(data=iris_data, orient="h")
plt.show()

# Split data para entrenamiento y validacion
X = iris_data.drop('species', axis=1)
y = iris_data['species']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Evaluacion de modelo
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")

# Comparar algoritmos
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# Predicciones de la validacion del dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluar predicciones
print(f"Accuracy score: {accuracy_score(Y_validation, predictions):.3f}")
print(f"Confusion matrix:\n{confusion_matrix(Y_validation, predictions)}\n")
print(f"Classification report:\n{classification_report(Y_validation, predictions)}")
