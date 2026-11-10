import numpy as np


class LogisticRegressionScratch:
    """Classificador de Regressão Logística vetorial implementado puramente com NumPy.

    Atributos:
        learning_rate (float): Taxa de aprendizado do gradiente descendente.
        iterations (int): Número máximo de épocas de treinamento.
        tol (float): Tolerância absoluta da variação de perda para parada antecipada.
        weights (np.ndarray | None): Vetor de pesos de forma (n_features,).
        bias (float): Termo de viés escalar.
        loss_history (list[float]): Histórico do custo (Binary Cross-Entropy) por época.
        converged_epoch (int | None): Época em que o critério de convergência foi atingido.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.learning_rate: float = learning_rate
        self.iterations: int = iterations
        self.tol: float = tol
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self.converged_epoch: int | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Calcula a ativação sigmoide com corte numérico contra overflow."""
        z_clipped = np.clip(z, -500.0, 500.0)
        return np.asarray(1.0 / (1.0 + np.exp(-z_clipped)), dtype=np.float64)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula a Binary Cross-Entropy garantindo estabilidade via epsilon."""
        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)
        m = y_true.shape[0]
        loss = -(1.0 / m) * np.sum(
            y_true * np.log(y_pred_safe) + (1.0 - y_true) * np.log(1.0 - y_pred_safe)
        )
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Treina o modelo via Gradiente Descendente em lote com Early Stopping.

        Args:
            X (np.ndarray): Matriz de features de forma (m_samples, n_features).
            y (np.ndarray): Vetor de rótulos binários de forma (m_samples,).

        Returns:
            self: O próprio estimador treinado.

        Raises:
            ValueError: Se as dimensões de X e y forem incompatíveis.
        """
        if X.ndim != 2:
            raise ValueError(f"X deve ter 2 dimensões (m_samples, n_features), recebido: {X.ndim}")
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(f"Incompatibilidade dimensional entre X {X.shape} e y {y.shape}")

        m_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []
        self.converged_epoch = None

        for epoch in range(1, self.iterations + 1):
            linear_output = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(linear_output)

            current_loss = self._compute_loss(y, y_hat)
            self.loss_history.append(current_loss)

            if len(self.loss_history) > 1:
                loss_delta = abs(self.loss_history[-1] - self.loss_history[-2])
                if loss_delta < self.tol:
                    self.converged_epoch = epoch
                    break

            error = y_hat - y
            dw = (1.0 / m_samples) * np.dot(X.T, error)
            db = (1.0 / m_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calcula probabilidades no intervalo [0, 1]."""
        if self.weights is None:
            raise RuntimeError("O modelo precisa ser treinado antes da inferência.")
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Gera predições binárias com base no limiar de decisão."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(np.int32)
