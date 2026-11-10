# Arquitetura e Engenharia do Sistema: Regressão Logística Vetorial

## 1. Visão Geral e Limites de Execução
O sistema consiste em um motor analítico desacoplado para aprendizado supervisionado binário, focado em estabilidade de ponto flutuante, vetorização e interpretabilidade através de uma interface interativa.

```mermaid
flowchart TD
    subgraph ClientLayer["Camada de Interface & UX"]
        UI["Dashboard Streamlit (app.py)"]
        Sliders["Hiperparâmetros & Controles Reativos"]
    end

    subgraph EvaluationLayer["Camada de Dados & Avaliação (evaluation.py)"]
        Gen["Gerador Gaussiano Sintético"]
        Split["train_test_split (80/20 Shuffle)"]
        Metrics["compute_classification_metrics (Matriz de Confusão, F1)"]
    end

    subgraph CoreEngine["Núcleo Numérico (logistic_regression.py)"]
        Model["LogisticRegressionScratch"]
        Sigmoid["Sigmoide com Proteção de Overflow [-500, 500]"]
        Loss["Binary Cross-Entropy (Log Loss com Epsilon)"]
        Optimizer["Otimizador Gradiente Descendente em Lote"]
        Stopping["Early Stopping (|J_t - J_{t-1}| < tol)"]
    end

    Sliders --> UI
    UI --> Gen --> Split
    Split -->|"X_train, y_train"| Model
    Model --> Sigmoid --> Loss --> Stopping
    Stopping -->|"Gradientes Analíticos"| Optimizer --> Model
    Split -->|"X_test, y_test"| Metrics
    Model -->|"Predições y_hat"| Metrics
    Metrics --> UI
```

---

## 2. Diagrama de Sequência: Ciclo de Otimização e Inferência

```mermaid
sequenceDiagram
    autonumber
    participant UI as Streamlit UI (app.py)
    participant Eval as Avaliação (evaluation.py)
    participant Core as Modelo (LogisticRegressionScratch)

    UI->>Eval: generate_synthetic_data(samples, std)
    Eval-->>UI: Matriz X, Vetor y
    UI->>Eval: train_test_split(X, y, test_size=0.2)
    Eval-->>UI: X_train, X_test, y_train, y_test

    UI->>Core: fit(X_train, y_train)
    loop Épocas de Treinamento (1 até iterations)
        Core->>Core: Forward Pass: z = Xw + b
        Core->>Core: Ativação: y_hat = sigma(z)
        Core->>Core: Cálculo de Custo J_t
        alt |J_t - J_{t-1}| < tol
            Core-->>Core: Dispara Early Stopping
        else Gradiente Normal
            Core->>Core: dw = (1/m) * X^T (y_hat - y)
            Core->>Core: db = (1/m) * sum(y_hat - y)
            Core->>Core: w = w - alpha * dw
        end
    end
    Core-->>UI: Modelo Otimizado (weights, bias)

    UI->>Core: predict(X_test, threshold)
    Core-->>UI: y_pred_test
    UI->>Eval: compute_classification_metrics(y_test, y_pred_test)
    Eval-->>UI: Métricas (Acurácia, Precisão, Recall, F1)
    UI-->>UI: Renderiza Fronteira de Decisão e Curva de Perda
```

---

## 3. Topologia e Modelo de Domínio (Entidades Numéricas)

```mermaid
classDiagram
    class LogisticRegressionScratch {
        +float learning_rate
        +int iterations
        +float tol
        +ndarray weights
        +float bias
        +list loss_history
        +int converged_epoch
        +fit(X: ndarray, y: ndarray) LogisticRegressionScratch
        +predict_proba(X: ndarray) ndarray
        +predict(X: ndarray, threshold: float) ndarray
        -_sigmoid(z: ndarray) ndarray
        -_compute_loss(y_true: ndarray, y_pred: ndarray) float
    }

    class EvaluationModule {
        +train_test_split(X, y, test_size, seed) tuple
        +compute_classification_metrics(y_true, y_pred) dict
    }

    class StreamlitDashboard {
        +generate_synthetic_data(samples, std, seed) tuple
        +render_plots(model, X_test, y_test)
    }

    StreamlitDashboard ..> EvaluationModule : Consome
    StreamlitDashboard ..> LogisticRegressionScratch : Orquestra
    EvaluationModule ..> LogisticRegressionScratch : Avalia
```

---

## 4. Referências Formais e Literatura
1. **Bishop, Christopher M.** (2006). *Pattern Recognition and Machine Learning*. Springer. Capítulo 4: Linear Models for Classification (Seção 4.3 - Probabilistic Discriminative Models).
2. **Hastie, Trevor; Tibshirani, Robert; Friedman, Jerome.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2ª Edição. Springer. Capítulo 4: Linear Methods for Classification.
3. **Goodfellow, Ian; Bengio, Yoshua; Courville, Aaron.** (2016). *Deep Learning*. MIT Press. Capítulo 5: Machine Learning Basics (Seção 5.7 - Supervised Learning Algorithms).
4. **IEEE 754-2019:** *Standard for Floating-Point Arithmetic*. Tratamento numérico de subtrações catastróficas, saturações assintóticas e prevenção de NaN/inf.
