import streamlit as st
import pandas as pd
import dill

# ==============================
# Cargar modelo entrenado
# ==============================
with open("RandomForest_dill.pkl","rb") as f:
    model = dill.load(f)

# Orden de columnas que el modelo espera
orden = ['Age',
        'Sex_F',
        'Sex_M',
        'RestingBP',
        'Cholesterol',
        'FastingBS',
        'MaxHR',
        'Oldpeak',
        'ST_Slope_Down',
        'ST_Slope_Flat',
        'ST_Slope_Up',
        'ChestPainType_ASY',
        'ChestPainType_ATA',
        'ChestPainType_NAP',
        'ChestPainType_TA',
        'ExerciseAngina_N',
        'ExerciseAngina_Y',
        'RestingECG_LVH',
        'RestingECG_Normal',
        'RestingECG_ST']

# ==============================
# Interfaz web con Streamlit
# ==============================
st.title("🫀 Clasificador de Enfermedades del Corazón")
st.write("Introduce los datos del paciente para predecir el riesgo.")

# Inputs numéricos
age = st.number_input("Edad", 20, 100, 50)
restingbp = st.number_input("Presión arterial en reposo (mmHg)", 80, 200, 120)
chol = st.number_input("Colesterol (mg/dl)", 100, 600, 200)
fastingbs = st.selectbox("Glucosa en ayunas > 120 mg/dl", [0, 1])  # 0 = No, 1 = Sí
maxhr = st.number_input("Frecuencia cardiaca máxima", 60, 220, 150)
oldpeak = st.number_input("Oldpeak (depresión ST)", -2.0, 6.0, 1.0, step=0.1)

# Variables categóricas
sex = st.selectbox("Sexo", ["F", "M"])
st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])
chest = st.selectbox("Tipo de dolor en el pecho", ["ASY", "ATA", "NAP", "TA"])
ex_angina = st.selectbox("Angina inducida por ejercicio", ["N", "Y"])
restingecg = st.selectbox("ECG en reposo", ["LVH", "Normal", "ST"])

# ==============================
# Procesar la entrada
# ==============================
if st.button("Predecir"):
    # Crear diccionario con one-hot encoding manual
    entrada = {
        'Age': age,
        'RestingBP': restingbp,
        'Cholesterol': chol,
        'FastingBS': fastingbs,
        'MaxHR': maxhr,
        'Oldpeak': oldpeak,
        'Sex_F': 1 if sex == "F" else 0,
        'Sex_M': 1 if sex == "M" else 0,
        'ST_Slope_Down': 1 if st_slope == "Down" else 0,
        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Up': 1 if st_slope == "Up" else 0,
        'ChestPainType_ASY': 1 if chest == "ASY" else 0,
        'ChestPainType_ATA': 1 if chest == "ATA" else 0,
        'ChestPainType_NAP': 1 if chest == "NAP" else 0,
        'ChestPainType_TA': 1 if chest == "TA" else 0,
        'ExerciseAngina_N': 1 if ex_angina == "N" else 0,
        'ExerciseAngina_Y': 1 if ex_angina == "Y" else 0,
        'RestingECG_LVH': 1 if restingecg == "LVH" else 0,
        'RestingECG_Normal': 1 if restingecg == "Normal" else 0,
        'RestingECG_ST': 1 if restingecg == "ST" else 0,
    }

    # Convertir a DataFrame y asegurar el orden correcto
    entrada_df = pd.DataFrame([entrada])
    entrada_df = entrada_df.reindex(columns=orden, fill_value=0)

    # ==============================
    # Predicción
    # ==============================
    pred = model.predict(entrada_df)

    # Mostrar resultado
    if pred[0] == 1:
        st.error("⚠️ Riesgo de enfermedad cardíaca")
    else:
        st.success("✅ Sin riesgo")
