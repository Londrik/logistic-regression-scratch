# Logistic Regression from Scratch (NumPy)

Implementação vetorial de ponta a ponta da Regressão Logística para classificação binária utilizando puramente NumPy e Python 3, acompanhada de um dashboard analítico em Streamlit.

## Arquitetura
- **Modelo:** `LogisticRegressionScratch` (`logistic_regression.py`)
- **Forward Pass:** Hipótese linear $z = Xw + b$ projetada via Sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$ com clamp numérico contra overflow.
- **Função de Custo:** Entropia Cruzada Binária (Log Loss) com tolerância $\epsilon = 10^{-15}$ contra indefinição de $\log(0)$.
- **Otimização:** Gradiente Descendente analítico vetorial $\nabla_w J = \frac{1}{m} X^T (\hat{y} - y)$.
- **Interface:** Streamlit (`app.py`) com cálculo dinâmico da reta da fronteira de decisão e métricas de matriz de confusão (TP, TN, FP, FN).

## Execução
```bash
streamlit run app.py

