import numpy as np
from typing import Tuple, Dict


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide arrays de features e labels em subconjuntos de treino e teste sem dependências externas."""
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size deve estar estritamente entre 0.0 e 1.0")

    rng = np.random.default_rng(seed)
    m_samples = X.shape[0]
    indices = np.arange(m_samples)
    rng.shuffle(indices)

    test_count = int(m_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float | int]:
    """Calcula matriz de confusão e métricas derivadas completas (Acurácia, Precisão, Recall, F1)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2 * (precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
