#VISUALIZACIONES 
#Primera visualizacion 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
df = pd.read_csv("Mental_Health_in_Tech_Survey.csv")

# sns.set_theme aplica un fondo blanco con cuadrícula para que el gráfico se vea mejor
sns.set_theme(style="whitegrid")


#PERSONAS QUE BUSCAN TRATAMIENTO VS LAS QUE NO 
print("Generando grafico 1...")
plt.figure(figsize=(8, 5)) #define el ancho y alto 
sns.countplot(data=df, x='treatment') #Cuenta automáticamente cuántas veces aparece cada categoría (yes/no) y dibuja las barras
plt.ylim(600, 650) 
plt.title("Búsqueda de tratamiento")
plt.show() # Muestra la ventana con el gráfico


#INTERFERENCIA DE UNA CONDICION CON SU DESEMPEÑO LABORAL 
print("Generando grafico 2...")
frecuencias = df['work_interfere'].value_counts()#agrupa y cuenta los valores 
frecuencias.plot.pie(autopct='%1.1f%%', figsize=(6, 6)) #toma las frecuencias y las grafica 
# autopct imprime el porcentaje con 1 decimal adentro de cada rebanada.
plt.title("Proporción de interferencia en el trabajo")
plt.show()


#EDADES DE LOS EMPLEADOS 
print("Generando grafico 3...")
edades_logicas = df[(df['Age'] >= 15) & (df['Age'] <= 100)] #el grafico solo usara edades que sean reales porque hay datos atipicos 
plt.figure(figsize=(8, 5))
# sns.histplot() crea un histograma agrupando las edades en rangos 
# kde=True dibuja una línea de tendencia suavizada sobre las barras.
sns.histplot(data=edades_logicas, x='Age', bins=20, kde=True)
plt.title("Distribución de edades")
plt.show()