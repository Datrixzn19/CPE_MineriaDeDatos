import pandas as pd
#Limpieza de nulos, duplicados, 
df = pd.read_csv("Mental_Health_in_Tech_Survey.csv")


df = df.drop_duplicates() #drop_duplicates() identifica filas idénticas y deja solo la primera aparición.

# Eliminamos los outliers
# Filtramos la tabla manteniendo solo las filas donde la Edad esté entre 15 y 100.
df = df[(df['Age'] >= 15) & (df['Age'] <= 100)]


# Tratamiento de nulos
# Recordemos que antes identificamos una columna con mas de 40% de nulos asi que la eliminamos.
df = df.drop('state', axis=1)# axis=1 indica que queremos eliminar una columna entera

# Esta columna tenia 1.4% de nulos. La corregimos con la moda.
moda_empleo = df['self_employed'].mode()[0]# mode()[0] extrae el valor de texto más frecuente de esa columna.
# fillna() busca los espacios vacíos y los rellena con el valor de la moda.
df['self_employed'] = df['self_employed'].fillna(moda_empleo)

# Esta columna tenia mas del 20% de nulos. Arreglamos con categoría neutra.
# No usamos la modamoda aquí, porque alteraríamos la distribución real de los datos.
df['work_interfere'] = df['work_interfere'].fillna("Don't know")




#Esto de aca es solo para local para trabajar entre archivos, no va en el cuaderno!
# index=False evita que Pandas cree una columna extra con los números de fila.
# (Nota para el cuaderno: Esta línea la puedes ELIMINAR cuando unas todo en un .ipynb)
df.to_csv("dataset_limpio.csv", index=False)

print("Limpieza completada y archivo guardado como 'dataset_limpio.csv'")