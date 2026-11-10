# ADR-002: Isolamento de Branches e Automação de Quality Gates Locais

- **Status:** Aceito
- **Data:** 2026-11-10
- **Decisores:** Henry Cardoso (@londrik), Architecture Guild

## Contexto
O projeto requer salvaguardas estritas para evitar alterações destrutivas ou não auditadas na branch de produção (main). Pushes contendo falhas de formatação, violações de tipo estático, riscos de segurança ou regressões matemáticas não devem atingir o repositório remoto.

## Decisão
1. **Branching Model:**
   - A branch main é exclusivamente protegida para versões estáveis.
   - O desenvolvimento ativo opera na branch developer ou em feature branches (feat/*, fix/*).
2. **Client-Side Hook (.git/hooks/pre-push):**
   - Interrompe a execução com código 1 se o usuário tentar disparar push direto para origin main.
   - Executa 4 etapas mandatórias em ordem:
     - Gate 1: Ruff (Linter & PEP 8).
     - Gate 2: Mypy (Type checking estrito).
     - Gate 3: Bandit (Análise de vulnerabilidades na AST).
     - Gate 4: Pytest (Validação determinística e testes numéricos).
3. **Resolução de Módulos Locais:**
   - Adição de pythonpath = ["."] na configuração do Pytest (pyproject.toml) para permitir resolução sem necessidade de builds intermediários ou pip install -e ..

## Consequências
- **Positivas:**
  - Garantia de que a branch main nunca quebra em produção.
  - Impossibilidade física de commit/push com testes quebrados no terminal local.
- **Negativas:**
  - Pushes locais levam alguns segundos a mais devido à execução sequencial dos linters e scanners.

## Alternativas Rejeitadas
- **Apenas CI Remoto (GitHub Actions):** Rejeitado porque permite poluir o histórico do Git com commits de "fix lint" ou "fix tests".
