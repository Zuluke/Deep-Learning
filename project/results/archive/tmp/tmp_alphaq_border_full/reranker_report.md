# AlphaTensor Structural Reranker

## Dataset

- Candidates: 47.
- Circuits: `cuccaro_adder_n3`, `gf_2pow2_mult`, `mod_5_4`, `qft_4`, `vbe_adder_3`.
- Label: `structural_cost`.
- Features are cheap QASM/decomposition statistics; ZX-derived target columns are not used as features.
- Predictions CSV: `results/tmp_alphaq_border_full/reranker_predictions.csv`.
- Evaluation CSV: `results/tmp_alphaq_border_full/reranker_eval.csv`.
- Comparison CSV: `results/tmp_alphaq_border_full/reranker_comparison.csv`.
- Baseline CSV: `results/tmp_alphaq_border_full/reranker_baselines.csv`.
- Model JSON: `results/tmp_alphaq_border_full/reranker_model.json`.
- Reranker selection summary: `results/tmp_alphaq_border_full/reranker_selection/public_resynth_summary.csv`.
- Prediction tolerance: 0.100.
- Ensemble size: 3.

## Leave-one-circuit-out evaluation

- Hit rate against true best candidate: 0.600.
- Mean primary regret vs true best: 0.313.
- Mean primary gain vs T-count-best: 0.109.
- `cuccaro_adder_n3`: selected `cuccaro_adder_n3:combo0`, true best `cuccaro_adder_n3:combo0`, regret=0.000, gain-vs-T-best=0.000.
- `gf_2pow2_mult`: selected `gf_2pow2_mult:combo8`, true best `gf_2pow2_mult:combo11`, regret=0.524, gain-vs-T-best=-0.333.
- `mod_5_4`: selected `mod_5_4:combo1`, true best `mod_5_4:combo1`, regret=0.000, gain-vs-T-best=0.000.
- `qft_4`: selected `qft_4:combo5`, true best `qft_4:combo14`, regret=1.041, gain-vs-T-best=0.000.
- `vbe_adder_3`: selected `vbe_adder_3:combo9`, true best `vbe_adder_3:combo9`, regret=0.000, gain-vs-T-best=0.880.

## Full-dataset fitted selections

- `cuccaro_adder_n3`: `cuccaro_adder_n3:combo0` pred=0.506, std=0.117, true=0.521, T=14.
- `gf_2pow2_mult`: `gf_2pow2_mult:combo11` pred=1.207, std=0.087, true=1.262, T=21.
- `mod_5_4`: `mod_5_4:combo1` pred=0.079, std=0.071, true=0.081, T=7.
- `qft_4`: `qft_4:combo6` pred=7.213, std=0.175, true=7.067, T=67.
- `vbe_adder_3`: `vbe_adder_3:combo9` pred=1.148, std=0.107, true=1.184, T=21.

## Selection comparison

- `cuccaro_adder_n3`: reranker `cuccaro_adder_n3:combo0` primary=0.521, T=14.0; structural-best `cuccaro_adder_n3:combo0` primary=0.521, T=14.0; T-best `cuccaro_adder_n3:combo0` primary=0.521, T=14.0.
- `gf_2pow2_mult`: reranker `gf_2pow2_mult:combo11` primary=1.262, T=21.0; structural-best `gf_2pow2_mult:combo11` primary=1.262, T=21.0; T-best `gf_2pow2_mult:combo7` primary=1.452, T=17.0.
- `mod_5_4`: reranker `mod_5_4:combo1` primary=0.081, T=7.0; structural-best `mod_5_4:combo1` primary=0.081, T=7.0; T-best `mod_5_4:combo1` primary=0.081, T=7.0.
- `qft_4`: reranker `qft_4:combo6` primary=7.067, T=67.0; structural-best `qft_4:combo14` primary=6.420, T=67.0; T-best `qft_4:combo5` primary=7.461, T=53.0.
- `vbe_adder_3`: reranker `vbe_adder_3:combo9` primary=1.184, T=21.0; structural-best `vbe_adder_3:combo9` primary=1.184, T=21.0; T-best `vbe_adder_3:combo2` primary=2.064, T=19.0.

## Objective baselines

- `qasm_depth`: mean regret=0.167, total T-delta=-4.
- `structural_oracle`: mean regret=0.000, total T-delta=0.
- `tcount`: mean regret=0.422, total T-delta=-20.
- `tdepth`: mean regret=0.033, total T-delta=0.
- `zx_total_depth`: mean regret=0.322, total T-delta=-18.

## Caveat

This is a small proof-of-concept dataset. Treat the model as a reranking probe for whether cheap features carry splitting signal, not as a mature general-purpose predictor yet.
