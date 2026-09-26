import pandas as pd
#Ingenieria de caracteristicas - Feature engineering 

df = pd.read_csv("dataset_transformado.csv") #Ignorar esta liena en el cuaderno

# Primera variable deribada - Índice de soporte corporativo 
# Sumamos si la empresa da beneficios Y si tiene programas de bienestar.
# .astype(int) convierte los valores booleanos generados por get_dummies a números de 0 o 1
# Esto nos da un rango de 0 (ningún soporte) a 2 (soporte total).
df['soporte_corporativo'] = df['benefits_Yes'].astype(int) + df['wellness_program_Yes'].astype(int)


# Segunda variable deribada - Bandera de interferencia alta 
# Si agrupamos categorías dispersas reduce el ruido y mejora la precisión.
# Si la interferencia Often o Sometimes le asignamos 1, caso contrario 0.
df['interferencia_alta'] = (df['work_interfere_Often'] | df['work_interfere_Sometimes']).astype(int)


df.to_csv("dataset_listo_modelos.csv", index=False)#Ignorar esta linea en el cuaderno 


print("Feature Engineering completado. Nuevas variables agregadas.")
print("Dimensiones finales para modelado:", df.shape)