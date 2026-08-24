import pandas as pd  

#Importamos el archivo csv 
df = pd.read_csv("Mental_Health_in_Tech_Survey.csv")

#Estructura del dataset 
print("ESTRUCTURA DE LA TABLA")
#Este comando nos resume las columnas, valores nulos y los tipos de datos
df.info()

print("DESCRIBE DE LA TABLA")
# 4. DETECCIÓN DE ANOMALÍAS EN VARIABLES NUMÉRICAS
print("\n--- ESTADÍSTICAS DE LA COLUMNA EDAD ---")
# df.describe() calcula métricas (conteo, media, min, max, desviación estándar) de columnas numéricas.
print(df['Age'].describe())
print("DESCRIBE DE LA TABLA 2")
print(df.describe())
print("ljfksd")