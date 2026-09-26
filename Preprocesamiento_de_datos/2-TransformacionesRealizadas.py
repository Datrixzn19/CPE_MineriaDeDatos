import pandas as pd
from sklearn.preprocessing import MinMaxScaler #pip install scikit-learn


df = pd.read_csv("dataset_limpio.csv") #Esta linea se omite en el cuaderno, se la uso para cargar el datasete del otro archivo 

# Codificacion de la variable objetivo
# map() busca las llaves del diccionario yes y no y las reemplaza por sus valores 1 o 0
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0}) #Se debe aislar la variable a predecir en formato binario.


# Normalizacio nde valores numericos
scaler = MinMaxScaler() #comprime los valores numéricos a un rango entre 0 y 1.
# Si la edad mínima es 18 y la máxima 70, 18 será 0.0 y 70 será 1.0. 
df[['Age']] = scaler.fit_transform(df[['Age']]) # Doble corchete porque fit_transform exige formato de DataFrame (2D)
# Esto evita que algoritmos basados en distancia (como KNN) le den más peso a la Edad que a otras variables.


# Codificacion de variables categoricas 
# pd.get_dummies() convierte todas las columnas de texto restantes en columnas binarias (0 y 1).
df_transformado = pd.get_dummies(df, drop_first=True)
# drop_first=True elimina la primera categoría de cada variable.

#Omitir estas dos lienas en el cuaderno 
df_transformado.to_csv("dataset_transformado.csv", index=False)
print("Transformaciones completadas. Dimensiones actuales:", df_transformado.shape)