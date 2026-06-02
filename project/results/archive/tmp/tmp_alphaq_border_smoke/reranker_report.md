# AlphaTensor Structural Reranker

## Dataset

- Candidates: 2.
- Circuits: `mod_5_4`.
- Label: `structural_cost`.
- Features are cheap QASM/decomposition statistics; ZX-derived target columns are not used as features.
- Predictions CSV: `results/tmp_alphaq_border_smoke/reranker_predictions.csv`.
- Evaluation CSV: `results/tmp_alphaq_border_smoke/reranker_eval.csv`.
- Comparison CSV: `results/tmp_alphaq_border_smoke/reranker_comparison.csv`.
- Baseline CSV: `results/tmp_alphaq_border_smoke/reranker_baselines.csv`.
- Model JSON: `results/tmp_alphaq_border_smoke/reranker_model.json`.
- Reranker selection summary: `results/tmp_alphaq_border_smoke/reranker_selection/public_resynth_summary.csv`.
- Prediction tolerance: 0.100.
- Ensemble size: 2.

## Leave-one-circuit-out evaluation

- Hit rate against true best candidate: 0.000.
- Mean primary regret vs true best: 0.000.
- Mean primary gain vs T-count-best: 0.000.

## Full-dataset fitted selections

- `mod_5_4`: `mod_5_4:combo1` pred=0.051, std=0.001, true=0.081, T=7.

## Selection comparison

- `mod_5_4`: reranker `mod_5_4:combo1` primary=0.081, T=7.0; structural-best `mod_5_4:combo1` primary=0.081, T=7.0; T-best `mod_5_4:combo1` primary=0.081, T=7.0.

## Objective baselines

- `qasm_depth`: mean regret=0.376, total T-delta=0.
- `structural_oracle`: mean regret=0.000, total T-delta=0.
- `tcount`: mean regret=0.000, total T-delta=0.
- `tdepth`: mean regret=0.000, total T-delta=0.
- `zx_total_depth`: mean regret=0.376, total T-delta=0.

## Caveat

This is a small proof-of-concept dataset. Treat the model as a reranking probe for whether cheap features carry splitting signal, not as a mature general-purpose predictor yet.
