import numpy as np
import pytest

from evaluation import compute_classification_metrics, train_test_split
from logistic_regression import LogisticRegressionScratch


def test_train_test_split_ratios() -> None:
    X = np.ones((100, 2))
    y = np.ones(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=1)
    assert X_train.shape == (80, 2)
    assert X_test.shape == (20, 2)
    assert y_train.shape == (80,)
    assert y_test.shape == (20,)


def test_sigmoid_overflow_resilience() -> None:
    large_values = np.array([-1000.0, 0.0, 1000.0])
    sig = LogisticRegressionScratch._sigmoid(large_values)
    assert not np.isnan(sig).any()
    assert not np.isinf(sig).any()
    assert sig[0] < 1e-10
    assert sig[1] == 0.5
    assert sig[2] > 0.999999


def test_model_convergence_separable_data() -> None:
    X = np.array([[-3.0, -3.0], [-2.0, -2.5], [2.0, 2.5], [3.0, 3.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegressionScratch(learning_rate=0.5, iterations=500, tol=1e-5)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)
    assert model.loss_history[-1] < model.loss_history[0]


def test_early_stopping_triggers() -> None:
    X = np.array([[-5.0, -5.0], [-4.0, -4.0], [4.0, 4.0], [5.0, 5.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegressionScratch(learning_rate=0.1, iterations=5000, tol=1e-3)
    model.fit(X, y)
    assert model.converged_epoch is not None
    assert model.converged_epoch < 5000
    assert len(model.loss_history) == model.converged_epoch


def test_classification_metrics_edge_cases() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5


def test_unfitted_model_raises_runtime_error() -> None:
    model = LogisticRegressionScratch()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0, 2.0]]))
