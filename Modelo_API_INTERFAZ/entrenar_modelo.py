# pip install pandas scikit-learn joblib kagglehub
import os
import joblib
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# kagglehub descarga el dataset directamente desde el repositorio oficial a un directorio temporal local.
path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")
# os.path.join concatena la ruta del directorio con el nombre exacto del archivo para evitar errores de barras (\ o /) según el sistema operativo.
df = pd.read_csv(os.path.join(path, "survey.csv"))

# df.drop_duplicates() escanea la matriz completa y elimina las filas donde el 100% de los valores coincidan con una fila anterior.
df = df.drop_duplicates()

# Filtrado booleano (Máscara): Se evalúa la condición interna devolviendo True o False. 
# Al pasarlo dentro de df[], Pandas recorta y conserva únicamente las filas que cumplen la condición geométrica (18 a 70).
df = df[(df['Age'] >= 18) & (df['Age'] <= 70)]

# fillna() busca los valores nulos (NaN o vacíos) exclusivamente en la serie 'work_interfere' y los rellena con un string neutro.
df['work_interfere'] = df['work_interfere'].fillna("Don't know")

# map() utiliza una estructura de diccionario {llave: valor} para buscar y reemplazar rápidamente datos. Vital para binarizar variables objetivo.
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

columnas_modelo = [
    'Age', 'family_history', 'work_interfere', 'remote_work',
    'benefits', 'care_options', 'wellness_program', 'anonymity', 'treatment'
]
# .copy() es crítico aquí: le indica a Pandas que reserve un espacio nuevo en la memoria RAM, evitando el warning SettingWithCopyWarning que surge al modificar fragmentos de un DataFrame antiguo.
df_subset = df[columnas_modelo].copy()

df_subset['family_history'] = df_subset['family_history'].map({'Yes': 1, 'No': 0})
df_subset['remote_work'] = df_subset['remote_work'].map({'Yes': 1, 'No': 0})
df_subset['anonymity'] = df_subset['anonymity'].map({'Yes': 1, 'No': 0, "Don't know": 2})

# get_dummies transforma variables de texto (nominales) en columnas numéricas (0 o 1).
# drop_first=True elimina estadísticamente la primera categoría (ej. Si hay Hombre/Mujer, deja solo Mujer: 1 es mujer, 0 implica que es hombre) para evitar colinealidad.
df_subset = pd.get_dummies(df_subset, columns=['work_interfere', 'benefits', 'care_options', 'wellness_program'], drop_first=True)

# axis=1 indica a Pandas que la operación de borrado aplica a una columna (eje Y). axis=0 aplicaría a una fila (eje X).
X = df_subset.drop('treatment', axis=1)
y = df_subset['treatment']

# .tolist() convierte el objeto Index nativo de Pandas en una lista básica de Python para poder guardarla posteriormente.
columnas_entrenamiento = X.columns.tolist()

# Función de scikit-learn que baraja y corta las matrices X (variables predictoras) e y (variable objetivo) manteniendo la alineación de los índices.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42)
# .fit() es el corazón de la minería de datos: desencadena los cálculos matemáticos para que el algoritmo extraiga patrones de los datos de entrenamiento.
modelo.fit(X_train, y_train)

# joblib.dump empaqueta un objeto de Python (en este caso un diccionario que contiene el árbol matemático y los nombres de las columnas) y lo convierte en un archivo físico binario (.pkl).
joblib.dump({'modelo': modelo, 'columnas': columnas_entrenamiento}, 'modelo_entrenado.pkl')
print("Modelo serializado y guardado exitosamente como 'modelo_entrenado.pkl'")