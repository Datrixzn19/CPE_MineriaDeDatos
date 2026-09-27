import pandas as pd
#pip install tabulate
from sklearn.model_selection import train_test_split # train_test_split divide la matriz de datos aleatoriamente
from sklearn.tree import DecisionTreeClassifier # DecisionTreeClassifier es el algoritmo de minería (Árbol de Decisión)
from sklearn.linear_model import LogisticRegression # LogisticRegression es el algoritmo lineal probabilístico
from sklearn.metrics import accuracy_score # accuracy_score evalúa qué tan bien predice el modelo

df = pd.read_csv("dataset_listo_modelos.csv")

# Separacion de variables X e y
X = df.drop('treatment', axis=1) # X (Características) - Todas las columnas excepto la variable objetivo.
y = df['treatment'] # y (Target) - La columna que el modelo debe aprender a predecir.

# Division de datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% para entrenamiento, 20% para prueba



# Primera tecnica - Arbol de decicion 
# criterion='gini' mide la calidad de la separación. max_depth=5 limita ramificaciones para evitar sobreajuste.
modelo_arbol = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

# .fit() dispara el entrenamiento. Aquí el algoritmo encuentra los patrones matemáticos.
modelo_arbol.fit(X_train, y_train)

# .predict() usa los datos de prueba, sin las respuestas, para generar sus pronósticos.
pred_arbol = modelo_arbol.predict(X_test)




# Segunda tecnica - Regresion logistica 
# solver='liblinear' es el optimizador matemático para clasificación. max_iter=1000 asegura que el algoritmo termine.
modelo_logistico = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
modelo_logistico.fit(X_train, y_train)
pred_logistico = modelo_logistico.predict(X_test)


# Comparacion de resultados 
# accuracy_score compara las predicciones del modelo con las respuestas reales 
acc_arbol = accuracy_score(y_test, pred_arbol)
acc_logistica = accuracy_score(y_test, pred_logistico)

# pd.DataFrame() construye una estructura tabular a partir de un diccionario para visualizar los datos
tabla_resultados = pd.DataFrame({
    'Tecnica de Mineria': ['Arbol de Decision', 'Regresion Logistica'],
    'Parametros': ['criterion="gini", max_depth=5', "solver='liblinear', max_iter=1000"],
    'Precision (Test)': [round(acc_arbol * 100, 2), round(acc_logistica * 100, 2)]
})


print("\nCOMPARACION DE MODELOS")
# .to_markdown() renderiza la tabla en la consola de manera legible
print(tabla_resultados.to_markdown(index=False))