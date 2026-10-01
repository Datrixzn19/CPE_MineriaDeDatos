#Este archivo contiene el codigo de la aplicacion web correspondiente al proyecto
#Esta disponible para su uso inmediato en el cuaderno de GoogleColab junto con el resto del codigo del proyecto: https://colab.research.google.com/drive/1tJU6-x6kD6SkfL8m3vx47i0PAftP-dII?usp=sharing#scrollTo=0Mpc0G4qEZlT

#Indicaciones para usarla de manera local 
"""
pip install fastapi uvicorn gradio joblib pydantic kagglehub scikit-learn pandas
Crear un archivo app.py colocar el codigo
"""

# Instalación de dependencias
# !pip install fastapi uvicorn gradio joblib pydantic kagglehub scikit-learn pandas

import os
import joblib
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
import gradio as gr
from pydantic import BaseModel
import uvicorn
import threading

print("🚀 Descargando dataset y entrenando modelo con múltiples variables...")

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO AVANZADO
# ==========================================
path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")
df = pd.read_csv(os.path.join(path, "survey.csv"))

# Limpieza básica de anomalías
df = df.drop_duplicates()
df = df[(df['Age'] >= 18) & (df['Age'] <= 70)]
df['work_interfere'] = df['work_interfere'].fillna("Don't know")
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

# Selección de variables clave para el modelo
columnas_modelo = [
    'Age', 'family_history', 'work_interfere', 'remote_work',
    'benefits', 'care_options', 'wellness_program', 'anonymity', 'treatment'
]
df_subset = df[columnas_modelo].copy()

# Codificación manual limpia para asegurar compatibilidad exacta
df_subset['family_history'] = df_subset['family_history'].map({'Yes': 1, 'No': 0})
df_subset['remote_work'] = df_subset['remote_work'].map({'Yes': 1, 'No': 0})
df_subset['anonymity'] = df_subset['anonymity'].map({'Yes': 1, 'No': 0, "Don't know": 2})

# One-hot encoding para variables categóricas
df_subset = pd.get_dummies(df_subset, columns=['work_interfere', 'benefits', 'care_options', 'wellness_program'], drop_first=True)

# Separación de X e y
X = df_subset.drop('treatment', axis=1)
y = df_subset['treatment']

columnas_entrenamiento = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del Árbol de Decisión
modelo = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42)
modelo.fit(X_train, y_train)

print("✅ Modelo entrenado con éxito.")


# ==========================================
# 2. CREACIÓN DE LA API (FASTAPI)
# ==========================================
app = FastAPI(title="API Predictiva de Salud Mental - Minería de Datos", version="2.0")

class EmployeeProfile(BaseModel):
    Age: int
    family_history: int  # 1: Sí, 0: No
    remote_work: int     # 1: Sí, 0: No
    anonymity: int       # 1: Sí, 0: No, 2: No sabe
    work_interfere: str  # "Often", "Rarely", "Sometimes", "Never", "Don't know"
    benefits: str        # "Yes", "No", "Don't know"
    care_options: str    # "Yes", "No", "Not sure"
    wellness_program: str # "Yes", "No", "Don't know"

@app.get("/")
def home():
    return {"estado": "API avanzada activa", "columnas_esperadas": columnas_entrenamiento}

@app.post("/predecir")
def predecir_api(data: EmployeeProfile):
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

    input_df = pd.DataFrame([input_dict])
    for col in columnas_entrenamiento:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columnas_entrenamiento]

    pred = int(modelo.predict(input_df)[0])
    prob = float(modelo.predict_proba(input_df)[0][1]) * 100

    diagnostico = "Alto requerimiento de tratamiento psicológico" if pred == 1 else "Baja probabilidad de requerir tratamiento"

    return {
        "prediccion": pred,
        "probabilidad_porcentaje": round(prob, 2),
        "diagnostico": diagnostico
    }


# ==========================================
# 3. CREACIÓN DE LA WEB APP (GRADIO) - CAMPOS VACÍOS POR DEFECTO
# ==========================================
def interfaz_web_completa(age, family_history, remote_work, anonymity, work_interfere, benefits, care_options, wellness_program):
    # Validación por si el usuario no llena algún campo y lo deja vacío
    if not family_history or not remote_work or not anonymity or not work_interfere or not benefits or not care_options or not wellness_program:
        return "⚠️ Por favor, complete todos los campos del formulario para realizar la evaluación predictiva."

    # Mapeo de valores en español a las categorías del modelo en inglés
    map_si_no = {"Sí": 1, "No": 0, "No sabe": 2, "No estoy seguro": 2}

    map_interfere = {
        "Frecuentemente": "Often",
        "Raramente": "Rarely",
        "A veces": "Sometimes",
        "Nunca": "Never",
        "No lo sé": "Don't know"
    }

    map_opciones = {
        "Sí": "Yes",
        "No": "No",
        "No lo sé": "Don't know",
        "No estoy seguro": "Not sure"
    }

    input_dict = {
        'Age': age,
        'family_history': map_si_no.get(family_history, 0),
        'remote_work': map_si_no.get(remote_work, 0),
        'anonymity': map_si_no.get(anonymity, 2)
    }

    df_temp = pd.DataFrame(columns=columnas_entrenamiento)
    df_temp.loc[0] = 0

    for k, v in input_dict.items():
        if k in df_temp.columns:
            df_temp.at[0, k] = v

    val_wi = map_interfere.get(work_interfere, "Don't know")
    col_wi = f"work_interfere_{val_wi}"
    if col_wi in df_temp.columns: df_temp.at[0, col_wi] = 1

    val_ben = map_opciones.get(benefits, "Don't know")
    col_ben = f"benefits_{val_ben}"
    if col_ben in df_temp.columns: df_temp.at[0, col_ben] = 1

    val_care = map_opciones.get(care_options, "Not sure")
    col_care = f"care_options_{val_care}"
    if col_care in df_temp.columns: df_temp.at[0, col_care] = 1

    val_well = map_opciones.get(wellness_program, "Don't know")
    col_well = f"wellness_program_{val_well}"
    if col_well in df_temp.columns: df_temp.at[0, col_well] = 1

    pred = modelo.predict(df_temp)[0]
    prob = modelo.predict_proba(df_temp)[0][1] * 100

    if pred == 1:
        return f"⚠️ ALERTA (Probabilidad: {prob:.1f}%): El perfil evaluado muestra indicadores significativos que sugieren la necesidad de buscar apoyo profesional o psicológico. Se recomienda activar protocolos de bienestar corporativo."
    else:
        return f"✅ ESTABLE (Probabilidad de riesgo: {prob:.1f}%): El perfil presenta baja probabilidad de requerir tratamiento médico o psicológico en este momento."

demo = gr.Interface(
    fn=interfaz_web_completa,
    inputs=[
        gr.Slider(minimum=18, maximum=70, value=30, step=1, label="Edad del empleado"),
        # Usamos value=None para que aparezcan vacíos por defecto y no haya confusión
        gr.Radio(["Sí", "No"], label="¿Tiene antecedentes familiares de enfermedad mental?", value=None),
        gr.Radio(["Sí", "No"], label="¿Trabaja de forma remota?", value=None),
        gr.Radio(["Sí", "No", "No sabe"], label="¿Su empresa garantiza anonimato al buscar ayuda?", value=None),
        gr.Dropdown(["Frecuentemente", "Raramente", "A veces", "Nunca", "No lo sé"], label="¿Con qué frecuencia su condición interfiere en el trabajo?", value=None),
        gr.Dropdown(["Sí", "No", "No lo sé"], label="¿Su empresa ofrece beneficios de salud mental?", value=None),
        gr.Dropdown(["Sí", "No", "No estoy seguro"], label="¿Conoce las opciones de cuidado de salud mental de su empresa?", value=None),
        gr.Dropdown(["Sí", "No", "No lo sé"], label="¿Su empresa ofrece programas de bienestar?", value=None)
    ],
    outputs=gr.Textbox(label="Evaluación Predictiva Institucional (Recursos Humanos)"),
    title="Sistema Inteligente de Predicción de Salud Mental en TI",
    description="Herramienta basada en Minería de Datos (Árbol de Decisión) en español. Seleccione una opción en cada campo para realizar la evaluación."
)


# ==========================================
# 4. EJECUCIÓN CONCURRENTE EN COLAB
# ==========================================
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

threading.Thread(target=run_fastapi, daemon=True).start()

print("🌐 API corriendo en: http://127.0.0.1:8000")
print("📖 Documentación Swagger interactiva: http://127.0.0.1:8000/docs")
print("\n🚀 Lanzando interfaz web interactiva en español...")

demo.launch(share=True, debug=False)