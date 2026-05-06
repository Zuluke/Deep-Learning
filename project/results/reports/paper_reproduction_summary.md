# Reproducao do artigo AlphaTensor-Quantum

## Cobertura

- Linhas comparadas com o artigo: 96.
- Matches exatos reproduzidos: 96/96.
- Linhas cujas decomposicoes selecionadas validam tensorialmente: 96/96.

## Familias

- `benchmarks_gadgets.npz`: 48 chaves de decomposicao, 366 candidatos, 366 candidatos tensor-equal.
- `benchmarks_no_gadgets.npz`: 48 chaves de decomposicao, 392 candidatos, 392 candidatos tensor-equal.
- `binary_addition.npz`: 8 chaves de decomposicao, 8 candidatos, 8 candidatos tensor-equal.
- `hamming_weight_phase_gradient.npz`: 17 chaves de decomposicao, 17 candidatos, 17 candidatos tensor-equal.
- `multiplication_finite_fields_gadgets.npz`: 9 chaves de decomposicao, 82 candidatos, 82 candidatos tensor-equal.
- `multiplication_finite_fields_no_gadgets.npz`: 9 chaves de decomposicao, 53 candidatos, 53 candidatos tensor-equal.
- `quantum_chemistry.npz`: 7 chaves de decomposicao, 60 candidatos, 60 candidatos tensor-equal.
- `unary_iteration_gadgets.npz`: 3 chaves de decomposicao, 30 candidatos, 30 candidatos tensor-equal.
- `unary_iteration_no_gadgets.npz`: 3 chaves de decomposicao, 30 candidatos, 30 candidatos tensor-equal.

## Figuras

- `benchmark_comparison`: `/Users/caio/Deep-Learning/project/results/reproducibility/paper/figures/paper_benchmark_comparison.png`
- `binary_addition`: `/Users/caio/Deep-Learning/project/results/reproducibility/paper/figures/binary_addition_effective_tcount.png`
- `family_coverage`: `/Users/caio/Deep-Learning/project/results/reproducibility/paper/figures/paper_family_coverage.png`
- `gf_multiplication`: `/Users/caio/Deep-Learning/project/results/reproducibility/paper/figures/gf_multiplication_effective_tcount.png`

## Interpretacao para a Entrega 1

Use esta reproducao como a replicacao principal do baseline. Ela valida as decomposicoes oficiais do AlphaTensor-Quantum recomputando seus tensores assinatura e recalculando os T-counts efetivos com gadgets reportados no artigo.

A reconstrucao QASM separada e as checagens com `feynver` continuam uteis como camada de auditoria, mas a verificacao tensorial e o alvo fiel de reproducao das decomposicoes otimizadas porque o artigo original publica decomposicoes otimizadas, nao circuitos reconstruidos completos para todos os casos.
