import pandas as pd
df = pd.read_csv("Mental_Health_in_Tech_Survey.csv")


cantidad_duplicados = df.duplicated().sum() #escanea el DataFrame fila por fila. 
# Retorna True solo si todos los valores de una fila son idénticos a los de una fila anterior.
# .sum() toma esa matriz de booleanos y suma los  True


# len(df) devuelve el número total de filas del DataFrame.
# se multiplica x100 para obtener el formato porcentual.
porcentaje_duplicados = (cantidad_duplicados / len(df)) * 100


print(f"Cantidad de duplicados: {cantidad_duplicados}")
print(f"Porcentaje: {porcentaje_duplicados:.2f}%")  #mostramos con dos decimales 