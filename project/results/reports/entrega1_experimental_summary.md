# Entrega 1 - Resultados experimentais preliminares

## Escopo executado

- Circuitos selecionados: `cuccaro_adder_n3`, `gf_2pow2_mult`, `hamming_15_low`, `mod_5_4`, `qcla_mod_7`, `qft_4`, `vbe_adder_3`.
- PyZX disponivel para 7/7 circuitos.
- AlphaTensor-public disponivel para 7/7 circuitos.
- Tabela consolidada: `/Users/caio/Deep-Learning/project/results/csv/entrega1_metrics.csv`.

## Figuras geradas

- ![T-count comparison](/Users/caio/Deep-Learning/project/results/figures/entrega1/tcount_comparison.png)
- ![Non-Clifford core comparison](/Users/caio/Deep-Learning/project/results/figures/entrega1/nonclifford_core_comparison.png)
- ![Reduction vs core area](/Users/caio/Deep-Learning/project/results/figures/entrega1/tcount_reduction_vs_core_area.png)

## Heatmaps de portas T/Tdg

- ![Heatmap t_gate_heatmap_mod_5_4](/Users/caio/Deep-Learning/project/results/figures/entrega1/t_gate_heatmap_mod_5_4.png)
- ![Heatmap t_gate_heatmap_qft_4](/Users/caio/Deep-Learning/project/results/figures/entrega1/t_gate_heatmap_qft_4.png)
- ![Heatmap t_gate_heatmap_vbe_adder_3](/Users/caio/Deep-Learning/project/results/figures/entrega1/t_gate_heatmap_vbe_adder_3.png)

## Principais resultados preliminares

- `qcla_mod_7` / AlphaTensor-public: T-count 413 -> 0 (delta=413, ganho relativo=1.000)
- `qcla_mod_7` / PyZX: T-count 413 -> 237 (delta=176, ganho relativo=0.426)
- `hamming_15_low` / AlphaTensor-public: T-count 161 -> 73 (delta=88, ganho relativo=0.547)
- `hamming_15_low` / PyZX: T-count 161 -> 97 (delta=64, ganho relativo=0.398)
- `vbe_adder_3` / AlphaTensor-public: T-count 70 -> 19 (delta=51, ganho relativo=0.729)
- `vbe_adder_3` / PyZX: T-count 70 -> 24 (delta=46, ganho relativo=0.657)
- `cuccaro_adder_n3` / AlphaTensor-public: T-count 42 -> 14 (delta=28, ganho relativo=0.667)
- `cuccaro_adder_n3` / PyZX: T-count 42 -> 16 (delta=26, ganho relativo=0.619)

## Reproducao fiel do artigo

- Linhas comparadas com o artigo: 96.
- Matches exatos de T-count efetivo: 96/96.
- Linhas com decomposicoes selecionadas validadas tensorialmente: 96/96.
- Fontes cobertas: `paper_fig4_binary_addition`, `paper_fig4_gf_gadgets`, `paper_fig4_gf_no_gadgets`, `paper_table_benchmark_gadgets`, `paper_table_benchmark_no_gadgets`.
- CSV de reproducao: `/Users/caio/Deep-Learning/project/results/reproducibility/paper/paper_benchmark_comparison.csv`.

## Verificacao formal

- Pares candidatos verificados: 34.
- Resultado: equal=21, inconclusive=5, timeout=8.
- CSV de verificacao: `/Users/caio/Deep-Learning/project/results/verification/entrega1/verification_summary.csv`.
- A tabela da Entrega 1 inclui `formal_verification_status` e `formal_proof_path` por metodo.

## Leitura para discussao

Os resultados ja permitem uma Entrega 1 experimental: PyZX reduz T-count em todos os casos selecionados em que ha ganho, enquanto o replay publico AlphaTensor frequentemente melhora PyZX nos circuitos com decomposicoes publicas compativeis. As figuras estruturais mostram que a reducao de T-count nem sempre coincide com uma reducao monotona do span temporal do nucleo nao-Clifford, o que sustenta a motivacao de medir estrutura e nao apenas contagem.

## Limitacoes documentadas

- O metodo `AlphaTensor-public` usa replay/ressintese de decomposicoes publicas, nao treino completo do artigo.
- A verificacao formal usa QASM normalizado para a base Clifford+T local; alguns circuitos ficam `inconclusive` ou `timeout` por limite do provador.
- As metricas de splitting sao proxies em lista de gates normalizada; ainda nao implementam a deteccao ZX-calculus do artigo de Clifford splitting.
- O T-depth reportado segue a implementacao local atual e deve ser tratado como estimativa simples para a Entrega 1.

## Proxima acao sugerida

Para escrever o draft, usar esta tabela e estas figuras nas secoes de Experimentos, Resultados preliminares e Discussao; a secao Solucao Proposta deve explicar que a contribuicao da Entrega 1 e a camada de analise estrutural sobre baselines publicos.
