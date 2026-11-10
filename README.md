# Logistic Regression from Scratch (NumPy)

Implementação vetorial da Regressão Logística para classificação binária do zero utilizando puramente NumPy e Python 3.11+, com dashboard interativo em Streamlit, Early Stopping, suite de métricas completas e esteira local/remota de Quality Gates.

## Arquitetura e Engenharia
- **Forward Pass:** $z = Xw + b$, mapeado via Sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$ com clamp numérico $[-500, 500]$.
- **Função de Custo:** Binary Cross-Entropy (Log Loss) com tolerância $\epsilon = 10^{-15}$.
- **Otimização:** Gradiente Descendente analítico vetorial $\nabla_w J = \frac{1}{m} X^T (\hat{y} - y)$.
- **Early Stopping:** Interrupção automática quando $|J_t - J_{t-1}| < \text{tol}$.
- **Avaliação:** Divisão de treino/teste independente (80/20) e cálculo de Matriz de Confusão, Acurácia, Precisão, Recall e F1-Score.

## Governança e Quality Gates
O repositório possui regras estritas configuradas no `.git/hooks/pre-push` e GitHub Actions:
1. **Ruff:** Verificação de PEP 8, formatação e imports.
2. **Mypy:** Checagem estática de tipos rígida (`--strict`).
3. **Bandit:** Varredura de segurança na AST do Python.
4. **Pytest:** Suíte de testes unitários matemáticos e de generalização.

## Como Executar

### 1. Iniciar o Dashboard Interativo
```bash
streamlit run app.py
```

### 2. Rodar a Suíte de Qualidade Manualmente
```bash
ruff check .
mypy logistic_regression.py evaluation.py
bandit -r . -x ./tests
pytest -v
```

## Documentação Técnica
- [Arquitetura de Software e Diagramas Mermaid](docs/ARCHITECTURE.md)
- [ADR-001: Implementação Numérica e Interface](docs/adr/ADR-001-implementacao-regressao-logistica-vetorial.md)
- [ADR-002: Governança de Branches e Quality Gates](docs/adr/ADR-002-git-flow-developer-e-prepush-gate.md)
