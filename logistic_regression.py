import numpy as np
from typing import List


class LogisticRegressionScratch:
    """Classificador de Regressão Logística vetorial implementado puramente com NumPy.

    Atributos:
        learning_rate (float): Taxa de aprendizado para o gradiente descendente.
        iterations (int): Número total de épocas de treinamento.
        weights (np.ndarray): Vetor de pesos de dimensão (n_features,).
        bias (float): Termo de viés escalar.
        loss_history (List[float]): Histórico do custo (Binary Cross-Entropy) por época.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000) -> None:
        self.learning_rate: float = learning_rate
        self.iterations: int = iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Aplica a função de ativação sigmoide com corte numérico contra overflow."""
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula a Binary Cross-Entropy com epsilon para prevenir log(0)."""
        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)
        m = y_true.shape[0]
        loss = -(1.0 / m) * np.sum(
            y_true * np.log(y_pred_safe) + (1.0 - y_true) * np.log(1.0 - y_pred_safe)
        )
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Executa a otimização por Gradiente Descendente em lote completo.

        Args:
            X (np.ndarray): Matriz de features de forma (m_samples, n_features).
            y (np.ndarray): Vetor de labels binários {0, 1} de forma (m_samples,).

        Returns:
            self: Instância ajustada do modelo.
        """
        m_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.iterations):
            linear_output = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(linear_output)

            current_loss = self._compute_loss(y, y_hat)
            self.loss_history.append(current_loss)

            error = y_hat - y
            dw = (1.0 / m_samples) * np.dot(X.T, error)
            db = (1.0 / m_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades calibradas no intervalo [0, 1]."""
        if self.weights is None:
            raise RuntimeError("O modelo precisa ser treinado antes da predição.")
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Classifica as entradas em {0, 1} com base em um limiar de corte."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(np.int32)
