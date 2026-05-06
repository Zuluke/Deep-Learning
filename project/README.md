# AlphaTensor-Quantum Repro Toolkit

Toolkit local para reproduzir resultados publicos do artigo **Quantum Circuit Optimization with AlphaTensor** e produzir a Entrega 1 do projeto “AlphaTensor-Quantum, Clifford Splitting e Hibridização Estrutural”.

O repositorio agora tem duas camadas complementares:

- **Reproducao fiel do artigo por tensor/decomposicao**: carrega as decomposicoes oficiais `.npz`, reconstrói os tensores assinatura, valida igualdade tensorial e recalcula o T-count efetivo com gadgets.
- **Auditoria experimental em QASM**: reconstrói circuitos quando localmente viavel, mede T-count/T-depth/depth, roda PyZX, gera métricas estruturais Clifford/non-Clifford, figuras e verificacao formal complementar com `feynver`.

Nao ha retreinamento completo do AlphaTensor-Quantum nesta entrega. A reproducao principal usa as decomposicoes publicadas pelo DeepMind, que sao o artefato adequado para comparar os valores reportados no artigo sem exigir o custo de treino original.

## Layout

- `src/atq_repro/`: APIs reutilizaveis para carregar decomposicoes, mapear tensores, validar igualdade tensorial, detectar gadgets e gerar resultados do paper.
- `scripts/`: wrappers CLI e pipelines de reproducao/auditoria.
- `notebooks/`: notebook executavel da Entrega 1.
- `external/`: snapshots vendorizados de `alphatensor_quantum`, `circuit-to-tensor` e ferramentas auxiliares.
- `results/csv/`: tabelas consolidadas.
- `results/figures/`: figuras para relatorio.
- `results/reproducibility/paper/`: CSVs/figuras da reproducao fiel do artigo.
- `results/reports/`: sumarios em Markdown para escrever a Entrega 1.

## Setup

```bash
cd /Users/caio/Deep-Learning/project
uv sync --group dev
```

Dependencias pesadas do demo upstream continuam separadas:

```bash
uv sync --group demo-cpu --group dev
```

## Reproducao fiel do artigo

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/reproduce_paper_results.py
```

Saidas principais:

- `results/reproducibility/paper/paper_results_long.csv`: uma linha por familia/decomposicao/candidato/bloco.
- `results/reproducibility/paper/paper_tensor_validation.csv`: igualdade tensorial para cada candidato.
- `results/reproducibility/paper/paper_benchmark_comparison.csv`: valores reproduzidos vs valores do artigo, com `match=true/false`.
- `results/reproducibility/paper/paper_family_summary.csv`: cobertura por familia.
- `results/reproducibility/paper/figures/`: figuras PNG/PDF de benchmarks, GF multiplication, binary addition e cobertura.
- `results/reports/paper_reproduction_summary.md`: sumario para a Entrega 1.

## Entrega 1 experimental

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/reproduce_paper_results.py
uv run python scripts/make_entrega1_outputs.py
```

Para uma auditoria completa do Draft 1, incluindo testes e notebook:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/run_draft1_pipeline.py --run-formal
```

O protocolo de reprodutibilidade esta documentado em `DRAFT1_REPRODUCIBILITY.md`.

Saidas principais:

- `results/csv/entrega1_metrics.csv`: tabela completa Original vs PyZX vs AlphaTensor-public.
- `results/csv/entrega1_metrics_formally_verified.csv`: tabela principal sem candidatos nao provados por `feynver`.
- `results/figures/entrega1/`: T-count, nucleo nao-Clifford, scatter e heatmaps.
- `results/figures/entrega1_formal/`: versao filtrada por verificacao formal.
- `results/reports/entrega1_experimental_summary.md`: narrativa curta com figuras, resultados e limitacoes.
- `results/reports/entrega1_formally_verified_summary.md`: resumo da tabela principal formalmente verificada.

## Notebook

O notebook de ponta a ponta fica em:

```text
notebooks/entrega1_reproducao_alphatensor_quantum.ipynb
```

Para executar do zero:

```bash
cd /Users/caio/Deep-Learning/project
uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/entrega1_reproducao_alphatensor_quantum.ipynb \
  --inplace \
  --ExecutePreprocessor.timeout=1200
```

Ele percorre o contexto do projeto, inventario dos dados locais, reproducao tensorial do paper, baselines praticos, metricas estruturais, figuras finais e discussao sobre por que o retreinamento completo fica fora do escopo da Entrega 1.

## Draft 1 em IEEE

O esboco do artigo fica fora desta subpasta, em:

```text
/Users/caio/Deep-Learning/paper/main.tex
/Users/caio/Deep-Learning/paper/main.pdf
```

O texto usa as tabelas e figuras geradas em `project/results/` e diferencia explicitamente a reproducao tensorial do artigo da auditoria QASM/formal.

## Verificacao formal complementar

Se `feynver` estiver disponivel, rode:

```bash
cd /Users/caio/Deep-Learning/project
uv run python scripts/run_formal_verification.py --scope entrega1 --timeout-sec 30
uv run python scripts/make_entrega1_outputs.py
```

Essa verificacao e uma auditoria em circuitos QASM reconstruidos. Para a reproducao dos valores do artigo, a validacao principal permanece sendo igualdade tensorial das decomposicoes oficiais.

## Testes

```bash
cd /Users/caio/Deep-Learning/project
uv run pytest tests -q
uv run python scripts/reproduce_paper_results.py
uv run jupyter nbconvert --to notebook --execute notebooks/entrega1_reproducao_alphatensor_quantum.ipynb --inplace --ExecutePreprocessor.timeout=1200
```

## Scripts legados ainda uteis

```bash
uv run python scripts/reproduce_fig4.py
uv run python scripts/reproduce_fig4b_binary_addition.py
uv run python scripts/replay_public_decompositions.py
```

Eles continuam funcionando como wrappers praticos, mas a logica reutilizavel da reproducao do artigo esta centralizada em `src/atq_repro/`.
