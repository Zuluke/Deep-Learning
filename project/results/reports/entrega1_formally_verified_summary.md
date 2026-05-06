# Entrega 1 - Tabela principal formalmente verificada

## Criterio de inclusao

Esta tabela remove candidatos com `timeout`, `inconclusive` ou qualquer outro status diferente de `equal` na verificacao formal. A tabela completa de auditoria continua preservada para documentar tudo que foi reproduzido.

- Circuitos com pelo menos um candidato formalmente verificado: `cuccaro_adder_n3`, `gf_2pow2_mult`, `mod_5_4`, `qft_4`, `vbe_adder_3`.
- Tabela principal sem timeouts: `/Users/caio/Deep-Learning/project/results/csv/entrega1_metrics_formally_verified.csv`.
- Tabela completa de auditoria: `/Users/caio/Deep-Learning/project/results/csv/entrega1_metrics.csv`.

## Figuras principais

- ![T-count comparison](/Users/caio/Deep-Learning/project/results/figures/entrega1_formal/tcount_comparison.png)
- ![Non-Clifford core comparison](/Users/caio/Deep-Learning/project/results/figures/entrega1_formal/nonclifford_core_comparison.png)
- ![Reduction vs core area](/Users/caio/Deep-Learning/project/results/figures/entrega1_formal/tcount_reduction_vs_core_area.png)

## Observacao para o texto

Os resultados de `hamming_15_low`, `qcla_mod_7` e candidatos AlphaTensor-public inconclusivos em circuitos como `cuccaro_adder_n3` e `vbe_adder_3` devem ser citados como reproducao experimental/auditoria, nao como tabela principal formalmente provada.
