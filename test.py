import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style='whitegrid')

# codigo para leer un csv 

df = pd.read_csv("C:/Users/PERSONAL/Desktop/repos/CPE_MineriaDeDatos/Mental_Health_in_Tech_Survey.csv")
    #Dispersion 
df_limpio = df[(df["Age"] >= 18) & (df["Age"] <= 100)]
plt.figure(figsize=(8, 5))

# sns.scatterplot es la función para puntos de dispersión
#las variables x y son las columnas que vayamos a comparar
sns.scatterplot(data=df_limpio, x='Age', y='Gender')
#estos comandos ya los vimos
plt.title('Relación entre Horas de Estudio y Calificación')
plt.xlabel('Horas de Estudio Diarias')
plt.ylabel('Puntuación General')
plt.show()
