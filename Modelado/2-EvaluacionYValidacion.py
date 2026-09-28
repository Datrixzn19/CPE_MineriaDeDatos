import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# CARGA DE DATOS Y SEPARACION DE VARIABLES
df = pd.read_csv("dataset_listo_modelos.csv")
X = df.drop('treatment', axis=1)
y = df['treatment']

# INSTANCIACION DE MODELOS (Usamos los mismos hiperparametros de la fase anterior)
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
modelo_logistico = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
modelo_arbol = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

print("Calculando validacion cruzada...")

# VALIDACION CRUZADA (CROSS-VALIDATION) 
# cross_val_score ejecuta el entrenamiento y prueba "cv" veces (5).
# scoring='accuracy' indica que queremos evaluar el porcentaje de aciertos.
cv_rf = cross_val_score(modelo_rf, X, y, cv=5, scoring='accuracy')
cv_logistico = cross_val_score(modelo_logistico, X, y, cv=5, scoring='accuracy')
cv_arbol = cross_val_score(modelo_arbol, X, y, cv=5, scoring='accuracy')

tabla_cv = pd.DataFrame({
    'Modelo': ['Random forest', 'Regresion logistica', 'Arbol de decision'],
    'Precision Media (CV=5)': [f"{cv_rf.mean()*100:.2f}%", f"{cv_logistico.mean()*100:.2f}%", f"{cv_arbol.mean()*100:.2f}%"]
})

print("\nRESULTADOS DE VALIDACION CRUZADA (K=5)")
print(tabla_cv.to_markdown(index=False))


# METRICAS AVANZADAS (SOBRE ARBOL DE DECISION) 
# Cambiamos al Arbol de Decision porque obtuvo el mejor rendimiento en tus pruebas.
# Hacemos una particion estandar (Hold-out) exclusivamente para extraer la matriz y el reporte.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Entrenamos y predecimos con el Arbol de Decision
modelo_arbol.fit(X_train, y_train)
predicciones = modelo_arbol.predict(X_test)
# confusion_matrix devuelve un array 2x2.
# .ravel() toma ese array de 2x2 y lo "aplana" en una sola dimension (4 valores seguidos)
# permitiendo asignarlos directamente a las variables tn, fp, fn, tp.
tn, fp, fn, tp = confusion_matrix(y_test, predicciones).ravel()

print("\nMATRIZ DE CONFUSION (Arbol de Decision)")
print(f"Verdaderos Positivos (TP): {tp} (Predijo Si, y era Si)")
print(f"Verdaderos Negativos (TN): {tn} (Predijo No, y era No)")
print(f"Falsos Positivos (FP): {fp} (Predijo Si, pero era No)")
print(f"Falsos Negativos (FN): {fn} (Predijo No, pero era Si)")

# classification_report calcula un resumen estadistico de Precision, Recall y F1-Score
reporte = classification_report(y_test, predicciones)
print("\nREPORTE DE CLASIFICACION")
print(reporte)



# VISUALIZACION DE RESULTADOS 
# Primer grafico - Barras para la validacion cruzada
# plt.figure crea una nueva ventana para el grafico.
plt.figure(figsize=(8, 5))
nombres_modelos = ['Random forest', 'Regresion logistica', 'Arbol de decision']
promedios_cv = [cv_rf.mean() * 100, cv_logistico.mean() * 100, cv_arbol.mean() * 100]

plt.bar(nombres_modelos, promedios_cv, color=['#4CAF50', '#2196F3', '#FFC107'])
plt.title('Precision Promedio por Modelo (Validacion Cruzada 5-Folds)')
plt.ylabel('Precision (%)')
# Forzamos el limite del eje Y de 80 a 85 para reflejar la escala de tu grafico anterior
plt.ylim(80, 85) 


# Segundo grafico - Mapa de calor para la matriz de confusion
plt.figure(figsize=(6, 5))
matriz_conf = confusion_matrix(y_test, predicciones)
# sns.heatmap aplica gradientes de color a una matriz. 
# annot=True escribe el valor, fmt='d' obliga a que el texto sea un numero entero (decimal integer).
sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No busca (0)', 'Busca (1)'],
            yticklabels=['No busca (0)', 'Busca (1)'])
plt.title('Matriz de Confusion - Arbol de Decision')
plt.xlabel('Prediccion del Modelo')
plt.ylabel('Realidad (y_test)')


plt.show() # mostramos las graficas