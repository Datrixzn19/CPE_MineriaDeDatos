# pip install gradio pandas joblib scikit-learn
import gradio as gr
import joblib
import pandas as pd

# Instanciación directa del modelo desde el almacenamiento local
datos_modelo = joblib.load('modelo_entrenado.pkl')
modelo = datos_modelo['modelo']
columnas_entrenamiento = datos_modelo['columnas']

def interfaz_web(age, family_history, remote_work, anonymity, work_interfere, benefits, care_options, wellness_program):
    # all() evalúa iterativamente una lista. Si algún valor es None, 0 o vacío, devuelve False.
    if not all([family_history, remote_work, anonymity, work_interfere, benefits, care_options, wellness_program]):
        return "⚠️ Complete todos los campos."

    map_si_no = {"Sí": 1, "No": 0, "No sabe": 2, "No estoy seguro": 2}
    map_interfere = {"Frecuentemente": "Often", "Raramente": "Rarely", "A veces": "Sometimes", "Nunca": "Never", "No lo sé": "Don't know"}
    map_opciones = {"Sí": "Yes", "No": "No", "No lo sé": "Don't know", "No estoy seguro": "Not sure"}

    input_dict = {
        'Age': age,
        'family_history': map_si_no.get(family_history, 0),
        'remote_work': map_si_no.get(remote_work, 0),
        'anonymity': map_si_no.get(anonymity, 2)
    }

    # Creación avanzada de matriz nula en Pandas: 
    # El argumento 'columns' pre-estructura la tabla pero la deja sin filas.
    df_temp = pd.DataFrame(columns=columnas_entrenamiento)
    
    # .loc[0] = 0 genera inmediatamente la primera fila (índice 0) propagando el valor 0 en cada una de las columnas transversales.
    df_temp.loc[0] = 0

    # Iteración sobre los ítems extraídos del diccionario
    for k, v in input_dict.items():
        if k in df_temp.columns:
            # .at[index, column] es un indexador escalar rápido (fastpath) exclusivo de Pandas. 
            # Es significativamente más eficiente que usar .loc para inyectar o modificar un valor único específico.
            df_temp.at[0, k] = v

    # Construcción dinámica de variables Dummy
    val_wi = map_interfere.get(work_interfere, "Don't know")
    if f"work_interfere_{val_wi}" in df_temp.columns: 
        df_temp.at[0, f"work_interfere_{val_wi}"] = 1

    val_ben = map_opciones.get(benefits, "Don't know")
    if f"benefits_{val_ben}" in df_temp.columns: 
        df_temp.at[0, f"benefits_{val_ben}"] = 1

    val_care = map_opciones.get(care_options, "Not sure")
    if f"care_options_{val_care}" in df_temp.columns: 
        df_temp.at[0, f"care_options_{val_care}"] = 1

    val_well = map_opciones.get(wellness_program, "Don't know")
    if f"wellness_program_{val_well}" in df_temp.columns: 
        df_temp.at[0, f"wellness_program_{val_well}"] = 1

    pred = modelo.predict(df_temp)[0]
    prob = modelo.predict_proba(df_temp)[0][1] * 100

    if pred == 1:
        return f"⚠️ ALERTA (Probabilidad: {prob:.1f}%)"
    return f"✅ ESTABLE (Probabilidad: {prob:.1f}%)"

# Objeto central de Gradio que empaqueta la función lógica de Python y levanta un servidor de sockets con HTML renderizado en tiempo real.
demo = gr.Interface(
    fn=interfaz_web,
    inputs=[
        gr.Slider(minimum=18, maximum=70, value=30, step=1, label="Edad"),
        gr.Radio(["Sí", "No"], label="Antecedentes familiares", value=None),
        gr.Radio(["Sí", "No"], label="Trabajo remoto", value=None),
        gr.Radio(["Sí", "No", "No sabe"], label="Anonimato", value=None),
        gr.Dropdown(["Frecuentemente", "Raramente", "A veces", "Nunca", "No lo sé"], label="Interferencia", value=None),
        gr.Dropdown(["Sí", "No", "No lo sé"], label="Beneficios", value=None),
        gr.Dropdown(["Sí", "No", "No estoy seguro"], label="Opciones de cuidado", value=None),
        gr.Dropdown(["Sí", "No", "No lo sé"], label="Programas de bienestar", value=None)
    ],
    outputs="text",
    title="Predicción de Salud Mental"
)

if __name__ == "__main__":
    demo.launch(share=True)