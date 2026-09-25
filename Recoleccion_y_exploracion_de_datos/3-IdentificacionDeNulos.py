import pandas as pd
df = pd.read_csv("Mental_Health_in_Tech_Survey.csv")
#IDENTIFICAMOS QUE VARIABLES TIENEN VALORES NULOS
#Una opcion mas sencilla es usar df.info como lo hicimos antes aunque esa no muestra porcentajes


#isnull() genera una matriz de nulos y no nulos
#sum() cuenta la cantidad de nulos por cada columna.
#Dividimos entre len(df) (total de filas, equivalente a COUNT(*)) y multiplicamos por 100.
porcentaje_nulos = (df.isnull().sum() / len(df)) * 100


# Solo mostramos las columnas cuyo porcentaje de nulos sea mayor a cero.
# .round(2) redondea el resultado a dos decimales 
columnas_con_nulos = porcentaje_nulos[porcentaje_nulos > 0].round(2)

print("PORCENTAJE DE NULOS POR VARIABLE")
print(columnas_con_nulos)