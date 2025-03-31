import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
# Cargar modelo
modelo = joblib.load("modelo_cancer.pkl")
# Cargar nombres de características desde el dataset original
data = load_breast_cancer()
features = data.feature_names
st.set_page_config(page_title="Clasificador de Cáncer de Mama", layout="wide")
st.title("🧬 Clasificación de Cáncer de Mama - ID Bootcamps")
st.write("Introduce los valores de las características para predecir si el tumor es **maligno** o **benigno**.")

# Crear sliders para TODAS las características (30)
inputs = []
cols = st.columns(3)  # Dividir en 3 columnas
for i in range(len(features)):
    col = cols[i % 3]
    with col:
        min_val = float(np.min(data.data[:, i]))
        max_val = float(np.max(data.data[:, i]))
        mean_val = float(np.mean(data.data[:, i]))
        if min_val == max_val:
            max_val += 1.0  # Evitar slider congelado
        val = st.slider(
            label=features[i],
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=0.01
        )
        inputs.append(val)

#FILTROS
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = maligno, 1 = benigno
# Combinar en un solo DataFrame
df = X.copy()
df["diagnóstico"] = y.map({0: "Maligno", 1: "Benigno"})
st.subheader("🔎 Exploración del dataset")
# Filtro por tipo de tumor
opcion_clase = st.selectbox("Filtrar por diagnóstico:", ["Todos", "Benigno", "Maligno"])
# Aplicar filtro
if opcion_clase != "Todos":
    df = df[df["diagnóstico"] == opcion_clase]