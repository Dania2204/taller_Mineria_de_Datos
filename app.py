from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Segmentación de clientes", page_icon="📊", layout="wide")

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
DATA_PATH = Path(__file__).with_name("Mall_Customers.csv")

st.title("📊 Segmentación de clientes con K-Means")
st.caption("Taller final de Minería de Datos | Dania Kathalina Urbina Portilla")
st.write(
    "Esta aplicación explora el dataset Mall Customers, compara valores de K y permite "
    "clasificar un cliente nuevo con ingreso anual y puntuación de gasto."
)

if not DATA_PATH.exists():
    st.error("No se encontró Mall_Customers.csv. Coloca el archivo en la misma carpeta que app.py.")
    st.stop()

df = pd.read_csv(DATA_PATH)
missing_features = [col for col in FEATURES if col not in df.columns]
if missing_features:
    st.error(f"Faltan columnas requeridas en el dataset: {missing_features}")
    st.stop()

X = df[FEATURES].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def perfil_cliente(ingreso, gasto):
    if ingreso >= 80 and gasto >= 60:
        return "Alto valor (ingreso y gasto altos)"
    if ingreso >= 80 and gasto < 40:
        return "Con potencial (ingreso alto, gasto bajo)"
    if ingreso < 40 and gasto >= 60:
        return "Gasto alto con presupuesto limitado"
    if ingreso < 40 and gasto < 40:
        return "Bajo valor actual"
    return "Cliente promedio o intermedio"

with st.sidebar:
    st.header("Configuración")
    k = st.slider("Número de clusters (K)", min_value=2, max_value=10, value=5, step=1)
    st.write("El valor inicial es K=5, seleccionado en el análisis del taller.")

inertias = []
ks_elbow = list(range(1, 11))
for candidate_k in ks_elbow:
    candidate_model = KMeans(n_clusters=candidate_k, random_state=42, n_init=10)
    candidate_model.fit(X_scaled)
    inertias.append(candidate_model.inertia_)

ks_sil = list(range(2, 11))
sil_values = []
for candidate_k in ks_sil:
    candidate_model = KMeans(n_clusters=candidate_k, random_state=42, n_init=10)
    candidate_labels = candidate_model.fit_predict(X_scaled)
    sil_values.append(silhouette_score(X_scaled, candidate_labels))

model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)
df_result = df.copy()
df_result["Cluster"] = labels
centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
centers.index.name = "Cluster"

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Clientes", f"{len(df):,}")
metric2.metric("K seleccionado", str(k))
metric3.metric("Silueta para K seleccionado", f"{sil_values[ks_sil.index(k)]:.3f}")

st.subheader("1. Evaluación del número de clusters")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks_elbow, inertias, marker="o")
    ax.axvline(k, linestyle="--", label=f"K seleccionado = {k}")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo")
    ax.set_xticks(ks_elbow)
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with chart_col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks_sil, sil_values, marker="o")
    ax.axvline(k, linestyle="--", label=f"K seleccionado = {k}")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Coeficiente de silueta")
    ax.set_title("Coeficiente de silueta")
    ax.set_xticks(ks_sil)
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("2. Clusters generados")
st.caption("Los identificadores numéricos son etiquetas del algoritmo; no representan un ranking.")
fig, ax = plt.subplots(figsize=(9, 5.5))
for cluster_id in sorted(df_result["Cluster"].unique()):
    group = df_result[df_result["Cluster"] == cluster_id]
    ax.scatter(group[FEATURES[0]], group[FEATURES[1]], alpha=0.75, label=f"Cluster {cluster_id}")
ax.scatter(centers[FEATURES[0]], centers[FEATURES[1]], marker="X", s=180, label="Centroides")
ax.set_xlabel("Ingreso anual (miles de dólares)")
ax.set_ylabel("Puntuación de gasto (1-100)")
ax.set_title(f"Segmentación K-Means con K={k}")
ax.grid(True, alpha=0.2)
ax.legend()
st.pyplot(fig)
plt.close(fig)

summary = df_result.groupby("Cluster").agg(
    Clientes=("CustomerID", "count"),
    Edad_promedio=("Age", "mean"),
    Ingreso_promedio=(FEATURES[0], "mean"),
    Gasto_promedio=(FEATURES[1], "mean"),
).round(2)
summary["Perfil orientativo"] = [
    perfil_cliente(float(centers.loc[cluster_id, FEATURES[0]]),
                   float(centers.loc[cluster_id, FEATURES[1]]))
    for cluster_id in summary.index
]
st.dataframe(summary, width="stretch")
st.write("Centroides en las unidades originales:")
st.dataframe(centers.round(2), width="stretch")

st.subheader("3. Clasificar un nuevo cliente")
st.write("Introduce valores para estimar a qué cluster se asignaría el cliente.")
input_col1, input_col2 = st.columns(2)
with input_col1:
    income = st.number_input(
        "Ingreso anual (miles de dólares)",
        min_value=0.0,
        max_value=200.0,
        value=60.0,
        step=1.0,
    )
with input_col2:
    spending = st.slider("Puntuación de gasto (1-100)", min_value=1, max_value=100, value=50)

if st.button("Clasificar cliente", type="primary"):
    new_customer = pd.DataFrame([{
        FEATURES[0]: float(income),
        FEATURES[1]: float(spending),
    }])
    new_scaled = scaler.transform(new_customer)
    predicted_cluster = int(model.predict(new_scaled)[0])
    centroid_income = float(centers.loc[predicted_cluster, FEATURES[0]])
    centroid_spending = float(centers.loc[predicted_cluster, FEATURES[1]])
    profile = perfil_cliente(centroid_income, centroid_spending)
    st.success(f"El cliente se asigna al Cluster {predicted_cluster}: {profile}.")
    st.write(
        f"Centroide aproximado del grupo: ingreso {centroid_income:.1f} k$ y "
        f"puntuación de gasto {centroid_spending:.1f}."
    )
    st.info(
        "La clasificación es una estimación basada en dos variables del dataset. "
        "No equivale a una predicción de ventas ni a una medición completa de rentabilidad."
    )

with st.expander("Ver una muestra del dataset"):
    st.dataframe(df.head(20), width="stretch")

st.divider()
st.caption(
    "Proyecto académico. Modelo: StandardScaler + KMeans(random_state=42, n_init=10). "
    "Los resultados deben validarse con datos comerciales adicionales antes de aplicarse a decisiones reales."
)
