# pip install fastapi uvicorn pydantic pandas joblib scikit-learn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="API Predictiva de Salud Mental")

# joblib.load lee el archivo binario del disco y reconstruye los objetos de Python tal como estaban en la memoria RAM antes de apagarse.
datos_modelo = joblib.load('modelo_entrenado.pkl')
modelo = datos_modelo['modelo']
columnas_entrenamiento = datos_modelo['columnas']

# Pydantic (BaseModel) impone un tipado estricto. Si una aplicación externa envía texto en 'Age', la API bloquea automáticamente la petición con un error HTTP 422 (Unprocessable Entity).
class EmployeeProfile(BaseModel):
    Age: int
    family_history: int  
    remote_work: int     
    anonymity: int       
    work_interfere: str  
    benefits: str        
    care_options: str    
    wellness_program: str 

@app.post("/predecir")
def predecir_api(data: EmployeeProfile):
    # Se simula el One-Hot Encoding al vuelo utilizando la concatenación de strings para igualar el formato matemático con el que aprendió el modelo.
    input_dict = {
        'Age': data.Age,
        'family_history': data.family_history,
        'remote_work': data.remote_work,
        'anonymity': data.anonymity,
        'work_interfere_' + data.work_interfere: 1 if data.work_interfere != 'Never' else 0,
        'benefits_' + data.benefits: 1 if data.benefits == 'Yes' else 0,
        'care_options_' + data.care_options: 1 if data.care_options == 'Yes' else 0,
        'wellness_program_' + data.wellness_program: 1 if data.wellness_program == 'Yes' else 0
    }

    # Se convierte el diccionario estructurado en un DataFrame de Pandas que contendrá una única fila.
    input_df = pd.DataFrame([input_dict])
    
    # Bucle de alineamiento: Scikit-learn arroja un error fatal si las columnas que se intentan predecir no coinciden exactamente con las de entrenamiento. 
    # Aquí agregamos las columnas ausentes y las inicializamos en 0.
    for col in columnas_entrenamiento:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reordenamos forzosamente la matriz de entrada basándonos en la lista oficial guardada en el archivo .pkl
    input_df = input_df[columnas_entrenamiento]

    # .predict() evalúa el DataFrame y devuelve un array de NumPy (ej. [1]). [0] extrae el escalar nativo de Python para que FastAPI lo pueda convertir en JSON.
    pred = int(modelo.predict(input_df)[0])
    # .predict_proba() devuelve probabilidades de todas las clases (ej. [0.15, 0.85]). La coordenada [0][1] extrae la certeza correspondiente a la clase 1 (busca ayuda).
    prob = float(modelo.predict_proba(input_df)[0][1]) * 100

    return {
        "prediccion": pred,
        "probabilidad_porcentaje": round(prob, 2)
    }

if __name__ == "__main__":
    # uvicorn arranca el servidor web ASGI. host="127.0.0.1" restringe la conexión solo a tu máquina local.
    uvicorn.run(app, host="127.0.0.1", port=8000)