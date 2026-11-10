import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressionScratch


def generate_synthetic_data(
    samples_per_class: int, cluster_std: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Gera dois agrupamentos 2D gaussianos linearmente separáveis."""
    rng = np.random.default_rng(seed)

    x0 = rng.normal(loc=-2.0, scale=cluster_std, size=(samples_per_class, 2))
    y0 = np.zeros(samples_per_class, dtype=np.int32)

    x1 = rng.normal(loc=2.0, scale=cluster_std, size=(samples_per_class, 2))
    y1 = np.ones(samples_per_class, dtype=np.int32)

    X = np.vstack((x0, x1))
    y = np.concatenate((y0, y1))
    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy}


st.set_page_config(page_title="Logistic Regression Lab", layout="wide")
st.title("Laboratório Interativo: Regressão Logística do Zero")

with st.sidebar:
    st.header("Hiperparâmetros & Dados")
    samples_per_class = st.slider("Amostras por classe", 50, 500, 150, step=25)
    cluster_std = st.slider("Dispersão dos dados (Std)", 0.5, 2.5, 1.2, step=0.1)
    learning_rate = st.select_slider(
        "Learning Rate (α)",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        value=0.1,
    )
    iterations = st.slider("Épocas de treino", 50, 3000, 500, step=50)
    decision_threshold = st.slider(
        "Limiar de Decisão (Threshold)", 0.1, 0.9, 0.5, step=0.05
    )

X, y = generate_synthetic_data(samples_per_class, cluster_std)
model = LogisticRegressionScratch(learning_rate=learning_rate, iterations=iterations)
model.fit(X, y)
y_pred = model.predict(X, threshold=decision_threshold)
metrics = compute_metrics(y, y_pred)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Acurácia", f"{metrics['accuracy'] * 100:.2f}%")
col_m2.metric("Verdadeiros Positivos (TP)", metrics["tp"])
col_m3.metric("Verdadeiros Negativos (TN)", metrics["tn"])
col_m4.metric("Falsos Positivos (FP)", metrics["fp"])
col_m5.metric("Falsos Negativos (FN)", metrics["fn"])

col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.subheader("Fronteira de Decisão (Plano Cartesiano)")
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="#3b82f6",
        label="Classe 0",
        alpha=0.7,
        edgecolors="k",
    )
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="#ef4444",
        label="Classe 1",
        alpha=0.7,
        edgecolors="k",
    )

    logit_val = float(np.log(decision_threshold / (1.0 - decision_threshold)))
    x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)

    if abs(model.weights[1]) > 1e-6:
        x2_vals = (logit_val - model.bias - model.weights[0] * x1_vals) / model.weights[
            1
        ]
        ax.plot(
            x1_vals,
            x2_vals,
            color="#10b981",
            linestyle="--",
            linewidth=2,
            label="Fronteira de Decisão",
        )

    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)

with col_plot2:
    st.subheader("Curva de Convergência (Binary Cross-Entropy)")
    fig_loss, ax_loss = plt.subplots(figsize=(7, 5))
    ax_loss.plot(
        range(len(model.loss_history)), model.loss_history, color="#6366f1", linewidth=2
    )
    ax_loss.set_xlabel("Épocas")
    ax_loss.set_ylabel("Custo $J(w, b)$")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig_loss)
