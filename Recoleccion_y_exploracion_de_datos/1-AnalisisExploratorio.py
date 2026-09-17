#ANALISIS EXPOLORATORIO DE DATOS 
import pandas as pd

df = pd.read_csv("Mental_Health_in_Tech_Survey.csv") #Llamamos al dataset que esta en nuestro proyecto de manera local 
pd.set_option('display.float_format', lambda x: '%.2f' % x) # pandas me estaba mostrando las estadisticas en el describe con un formato raro, con esta linea lo solucione(corregir esto al final!!!)

print("Estadisticas descriptivas basicas")
print("df.info")#Nos muestra todas las variables, si hay valores nulos, los tipos de datos de cada variable y el tamaño en KB del dataset 
print(df.info())

print("df.describe")#Analiza variables numericas, muestra informacion como, valores maximos, minimos, percentiles, etc.
print(df.describe())

print("df.head(n)")##Nos muestra una parte de los datos para que podamos ver el contenido de las variables, podemos para saber como es la estructura del dataset
print(df.head(10))

