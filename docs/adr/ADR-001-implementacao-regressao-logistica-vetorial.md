# ADR-001: Implementação Vetorial da Regressão Logística Sem Frameworks de Alto Nível

- **Status:** Aceito
- **Data:** 2026-11-09
- **Decisores:** Henry Cardoso (@londrik), Architecture Guild

## Contexto
Para validar fundamentos de álgebra linear e otimização numérica aplicados a aprendizado de máquina, faz-se necessário implementar o pipeline de classificação binária diretamente a partir dos princípios matemáticos, desacoplado de dependências pesadas (como Scikit-Learn ou PyTorch). O sistema exige visualização gráfica em tempo real dos pesos e da fronteira de decisão.

## Decisão
1. **Núcleo Matemático:** Implementado na classe `LogisticRegressionScratch` em NumPy puro de 64 bits float, com operações matriciais vetorizadas via BLAS/LAPACK.
2. **Mitigação de Exceções de Ponto Flutuante:** 
   - Saturação da Sigmoide via `np.clip(z, -500.0, 500.0)`.
   - Limitação assintótica do argumento logarítmico do Log Loss com tolerância $\epsilon = 10^{-15}$.
3. **Critério de Eficiência Computacional:** Interrupção do ciclo de gradiente descendente por Early Stopping baseado em tolerância absoluta de perda ($|J_t - J_{t-1}| < \text{tol}$).
4. **Interface Gráfica:** Adoção de Streamlit pela reatividade declarativa nativa com tensores NumPy e Matplotlib.

## Consequências
- **Positivas:** 
  - Ausência de dependências proprietárias ou caixas-pretas de otimização.
  - Zero alocação desnecessária de tensores em memória RAM.
  - Testabilidade unitária estrita e determinística.
- **Negativas:** 
  - Otimização restrita à CPU (sem aceleração automática por CUDA/GPU).
  - Suporte limitado a problemas estritamente lineares no espaço das features fornecido.

## Alternativas Rejeitadas
- **Scikit-Learn (`LogisticRegression`):** Rejeitado por mascarar a derivação analítica dos gradientes e o vetor de erros.
- **PyTorch / JAX:** Rejeitado pela sobrecarga desnecessária de grafos computacionais e runtime compilation para um modelo analítico de apenas dois parâmetros ($w, b$).
