import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from evaluation import compute_classification_metrics, train_test_split
from logistic_regression import LogisticRegressionScratch


def generate_synthetic_data(
    samples_per_class: int, cluster_std: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=-2.0, scale=cluster_std, size=(samples_per_class, 2))
    y0 = np.zeros(samples_per_class, dtype=np.int32)
    x1 = rng.normal(loc=2.0, scale=cluster_std, size=(samples_per_class, 2))
    y1 = np.ones(samples_per_class, dtype=np.int32)
    return np.vstack((x0, x1)), np.concatenate((y0, y1))


st.set_page_config(page_title="Logistic Regression Lab", layout="wide")
st.title("Laboratório de Otimização: Regressão Logística Vetorial")

with st.sidebar:
    st.header("1. Geração de Dados")
    samples_per_class = st.slider("Amostras por classe", 50, 1000, 200, step=25)
    cluster_std = st.slider("Dispersão dos dados (Std)", 0.5, 3.5, 1.2, step=0.1)
    test_size = st.slider("Divisão Teste (%)", 10, 40, 20, step=5) / 100.0

    st.header("2. Hiperparâmetros de Treino")
    learning_rate = st.select_slider(
        "Taxa de Aprendizado (α)",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        value=0.1,
    )
    iterations = st.slider("Épocas Máximas", 50, 5000, 1000, step=50)
    tol = st.select_slider(
        "Tolerância Early Stopping (|ΔJ|)",
        options=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.0],
        value=1e-5,
    )
    threshold = st.slider("Limiar de Decisão (Threshold)", 0.05, 0.95, 0.5, step=0.05)

X, y = generate_synthetic_data(samples_per_class, cluster_std)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

model = LogisticRegressionScratch(learning_rate=learning_rate, iterations=iterations, tol=tol)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train, threshold=threshold)
y_pred_test = model.predict(X_test, threshold=threshold)

train_metrics = compute_classification_metrics(y_train, y_pred_train)
test_metrics = compute_classification_metrics(y_test, y_pred_test)

status_text = (
    f"Convergido na época {model.converged_epoch}"
    if model.converged_epoch
    else f"Completou todas as {iterations} épocas"
)
st.info(
    f"Status do Otimizador: **{status_text}** | Conjunto de Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras"
)

st.markdown("### Métricas de Generalização (Avaliação em Teste)")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Acurácia",
    f"{test_metrics['accuracy'] * 100:.2f}%",
    delta=f"{(test_metrics['accuracy'] - train_metrics['accuracy']) * 100:.2f}% vs Treino",
)
c2.metric("Precisão", f"{test_metrics['precision'] * 100:.2f}%")
c3.metric("Recall (Sensibilidade)", f"{test_metrics['recall'] * 100:.2f}%")
c4.metric("F1-Score", f"{test_metrics['f1_score']:.4f}")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.subheader("Fronteira de Decisão no Conjunto de Teste")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        X_test[y_test == 0, 0],
        X_test[y_test == 0, 1],
        color="#3b82f6",
        label="Classe 0 (Teste)",
        alpha=0.7,
        edgecolors="k",
    )
    ax.scatter(
        X_test[y_test == 1, 0],
        X_test[y_test == 1, 1],
        color="#ef4444",
        label="Classe 1 (Teste)",
        alpha=0.7,
        edgecolors="k",
    )

    logit_val = float(np.log(threshold / (1.0 - threshold)))
    x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    if abs(model.weights[1]) > 1e-6:
        x2_vals = (logit_val - model.bias - model.weights[0] * x1_vals) / model.weights[1]
        ax.plot(
            x1_vals,
            x2_vals,
            color="#10b981",
            linestyle="--",
            linewidth=2,
            label="Fronteira",
        )

    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)

with col_p2:
    st.subheader("Curva de Perda (Binary Cross-Entropy)")
    fig_l, ax_l = plt.subplots(figsize=(7, 5))
    ax_l.plot(
        range(1, len(model.loss_history) + 1),
        model.loss_history,
        color="#6366f1",
        linewidth=2,
    )
    if model.converged_epoch:
        ax_l.axvline(
            x=model.converged_epoch,
            color="#f59e0b",
            linestyle=":",
            label=f"Parada (Época {model.converged_epoch})",
        )
    ax_l.set_xlabel("Épocas")
    ax_l.set_ylabel("Custo $J(w, b)$")
    ax_l.legend()
    ax_l.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig_l)
