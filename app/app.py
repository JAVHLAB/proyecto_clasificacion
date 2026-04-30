import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.predict import predict
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

st.set_page_config(page_title="Predicción de Préstamo", layout="centered")
st.title("Predicción de Aprobación de Préstamo")

st.header("Datos del solicitante")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad (Age)", min_value=18, max_value=100, value=30)
    income = st.number_input("Ingreso (Income)", min_value=0, value=50000)
    loan_amount = st.number_input("Monto del préstamo (LoanAmount)", min_value=0, value=10000)
    credit_score = st.number_input("Puntaje de crédito (CreditScore)", min_value=300, max_value=850, value=650)
    years_experience = st.number_input("Años de experiencia (YearsExperience)", min_value=0, max_value=50, value=5)

with col2:
    gender = st.selectbox("Género (Gender)", ["Male", "Female", "Other"])
    education = st.selectbox("Educación (Education)", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Tipo de empleo (EmploymentType)", ["Salaried", "Self-Employed", "Unemployed"])
    city = st.selectbox("Ciudad (City)", ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])

st.divider()

if st.button("Predecir", use_container_width=True):
    input_data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "YearsExperience": years_experience,
        "Gender": gender,
        "Education": education,
        "EmploymentType": employment_type,
        "City": city,
        "LoanApproved": 0
    }

    try:
        result = predict(input_data)

        st.subheader("Resultado de la predicción")
        if result == 1:
            st.success("✅ Préstamo APROBADO")
        else:
            st.error("❌ Préstamo NO APROBADO")
    except Exception as e:
        st.error(f"Error al predecir: {e}")

st.divider()
st.header("Métricas del modelo")

@st.cache_data
def get_metrics():
    df = load_data("datos/raw/loan_risk_prediction_dataset.csv")
    df = preprocess_data(df)
    X = df.drop("LoanApproved", axis=1)
    y = df["LoanApproved"]
    model = joblib.load("models/decision_tree_model.pkl")
    _, X_test, y_test = train_model(X, y)
    acc, cm = evaluate_model(model, X_test, y_test)
    return acc, cm

try:
    acc, cm = get_metrics()

    st.metric("Porcentaje de eficiencia", f"{acc * 100:.2f}%")

    st.subheader("Matriz de confusión")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Aprobado", "Aprobado"],
                yticklabels=["No Aprobado", "Aprobado"], ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

except Exception as e:
    st.warning(f"No se pudieron cargar las métricas: {e}")
