# AlphaTensor-Quantum Bootstrap

Bootstrap reproduzível da Fase 1 descrita em [`/Users/caio/Deep-Learning/main.tex`](/Users/caio/Deep-Learning/main.tex).

## Objetivo desta fase

- preservar `main.tex` e o worktree principal;
- vendorizar `alphatensor_quantum` e `circuit-to-tensor` sem patch local;
- fixar o ambiente com `uv` em Python `3.11`;
- validar o demo público em perfil `cpu` para smoke tests;
- preparar o perfil `cuda` para a execução oficial em Linux/A100;
- comprovar um pipeline mínimo `compile/resynth`.

## Layout

- `external/`: snapshots vendorizados dos repositórios upstream.
- `configs/`: reservado para configs futuras.
- `data/`: reservado para circuitos e artefatos de entrada futuros.
- `docs/`: notas operacionais desta fase.
- `results/`: logs, saídas do pipeline e manifesto de reprodutibilidade.
- `scripts/`: wrappers e utilitários locais.

## Comandos principais

```bash
cd /Users/caio/Deep-Learning/project
./scripts/setup_env.sh --profile cpu
uv run python scripts/run_demo_train.py --mode smoke --profile cpu --log-dir results/logs/demo
./scripts/run_compile_pipeline.sh --input external/circuit-to-tensor/examples/mod_5_4.qasm --output-dir results/compile/mod_5_4 --zx-preopt off
./scripts/run_resynth.sh --decomposition results/compile/mod_5_4/mod_5_4.block1.matrix.npy --mapping results/compile/mod_5_4/mod_5_4.block1.mapping.txt --original results/compile/mod_5_4/mod_5_4.block1.matrix.npy --output-dir results/resynth/mod_5_4 --gadgets off
```

## Observações

- O demo oficial continua sendo o entrypoint upstream `python -m alphatensor_quantum.src.demo.run_demo`.
- O perfil `cpu` existe para bootstrap e smoke tests locais; a validação oficial do demo fica reservada ao perfil `cuda` em Linux/A100.
- `feynver` continua opcional nesta fase; sem ele, o status de verificação deve permanecer `not-run`.
