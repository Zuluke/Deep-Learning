# Entrega 1 - Resultados experimentais preliminares

## Escopo executado

- Circuitos selecionados: `gf_2pow2_mult`, `hamming_weight_n4`, `hamming_weight_n5`, `mod_5_4`, `qft_4`.
- PyZX disponivel para 5/5 circuitos.
- AlphaTensor-public disponivel para 5/5 circuitos.
- AlphaQ tensor-v3 disponivel para 5/5 circuitos.
- AlphaQ tensor-v3 phase-slack disponivel para 5/5 circuitos.
- AlphaQ-final disponivel para 5/5 circuitos.
- Tabela consolidada: `/Users/caio/Deep-Learning/project/results/csv/entrega1_metrics.csv`.

## Figuras geradas

- ![T-count comparison](/Users/caio/Deep-Learning/project/results/figures/entrega1/tcount_comparison.png)
- ![Non-Clifford core comparison](/Users/caio/Deep-Learning/project/results/figures/entrega1/nonclifford_core_comparison.png)
- ![ZX splitting comparison](/Users/caio/Deep-Learning/project/results/figures/entrega1/zx_splitting_comparison.png)
- ![Reduction vs core area](/Users/caio/Deep-Learning/project/results/figures/entrega1/tcount_reduction_vs_core_area.png)

## Heatmaps de portas T/Tdg

- ![Heatmap t_gate_heatmap_mod_5_4](/Users/caio/Deep-Learning/project/results/figures/entrega1/t_gate_heatmap_mod_5_4.png)
- ![Heatmap t_gate_heatmap_qft_4](/Users/caio/Deep-Learning/project/results/figures/entrega1/t_gate_heatmap_qft_4.png)
- ![Heatmap t_gate_heatmap_hamming_weight_n4](/Users/caio/Deep-Learning/project/results/figures/entrega1/t_gate_heatmap_hamming_weight_n4.png)

## Principais resultados preliminares

- `mod_5_4` / AlphaQ-final: T-count 28 -> 7 (delta=21, ganho relativo=0.750)
- `mod_5_4` / AlphaQ tensor-v3: T-count 28 -> 7 (delta=21, ganho relativo=0.750)
- `mod_5_4` / AlphaQ tensor-v3 phase-slack: T-count 28 -> 7 (delta=21, ganho relativo=0.750)
- `mod_5_4` / AlphaTensor-public: T-count 28 -> 7 (delta=21, ganho relativo=0.750)
- `mod_5_4` / PyZX: T-count 28 -> 8 (delta=20, ganho relativo=0.714)
- `qft_4` / AlphaQ tensor-v3: T-count 69 -> 53 (delta=16, ganho relativo=0.232)
- `qft_4` / AlphaQ tensor-v3 phase-slack: T-count 69 -> 53 (delta=16, ganho relativo=0.232)
- `qft_4` / AlphaTensor-public: T-count 69 -> 53 (delta=16, ganho relativo=0.232)

## Reproducao fiel do artigo

- Linhas comparadas com o artigo: 96.
- Matches exatos de T-count efetivo: 96/96.
- Linhas com decomposicoes selecionadas validadas tensorialmente: 96/96.
- Fontes cobertas: `paper_fig4_binary_addition`, `paper_fig4_gf_gadgets`, `paper_fig4_gf_no_gadgets`, `paper_table_benchmark_gadgets`, `paper_table_benchmark_no_gadgets`.
- CSV de reproducao: `/Users/caio/Deep-Learning/project/results/reproducibility/paper/paper_benchmark_comparison.csv`.

## Verificacao formal

- Pares candidatos verificados: 69.
- Resultado: equal=52, inconclusive=9, timeout=8.
- CSV de verificacao: `/Users/caio/Deep-Learning/project/results/verification/entrega1/verification_summary.csv`.
- A tabela da Entrega 1 inclui `formal_verification_status` e `formal_proof_path` por metodo.

## Leitura para discussao

Os resultados ja permitem uma Entrega 1 experimental: PyZX reduz T-count em todos os casos selecionados em que ha ganho, enquanto o replay publico AlphaTensor frequentemente melhora PyZX nos circuitos com decomposicoes publicas compativeis. A variante AlphaQ tensor-v3 aparece como selecao AlphaQuantum-only orientada pela geometria tensorial, sem usar ZX/feynver no loop de escolha. As figuras estruturais e a fronteira detectada em ZX mostram que a reducao de T-count nem sempre coincide com uma melhora monotona da separacao Clifford/nao-Clifford, o que sustenta a motivacao de medir estrutura e nao apenas contagem.

## Limitacoes documentadas

- O metodo `AlphaTensor-public` usa replay/ressintese de decomposicoes publicas, nao treino completo do artigo.
- A verificacao formal usa QASM normalizado para a base Clifford+T local; candidatos `inconclusive` ou `timeout` ficam fora da tabela principal.
- A deteccao ZX implementada e conservadora: usa a fronteira em diagramas circuit-like e fecha recursivamente portas de dois qubits que cruzam a separacao.
- O T-depth reportado segue a implementacao local atual e deve ser tratado como estimativa simples para a Entrega 1.

## Proxima acao sugerida

Para escrever o draft, usar esta tabela e estas figuras nas secoes de Experimentos, Resultados preliminares e Discussao; a secao Solucao Proposta deve explicar que a contribuicao da Entrega 1 e a camada de analise estrutural sobre baselines publicos.
