import pandas as pd
df = pd.read_csv("Mental_Health_in_Tech_Survey.csv")

#Limites reales que puede rondar un trabajador del sector tech 
limite_inferior = 15
limite_superior = 100

atipicos = df[(df['Age'] < limite_inferior) | (df['Age'] > limite_superior)]# Trae las filas donde la Edad sea MENOR a 15 O MAYOR a 100".

cantidad_atipicos = len(atipicos) #cuenta la cantidad de filas del DataFrame 'atipicos'

print(f"Cantidad de valores atípicos detectados: {cantidad_atipicos}")
print("Valores exactos encontrados:")
print(atipicos['Age'].values)#extrae solo los números en forma de lista, ignorando el resto de las columnas