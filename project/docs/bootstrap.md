# Notas Operacionais

## Estratégia de vendor

Os snapshots em `external/` foram materializados a partir de uma branch auxiliar baseada em `git subtree`, sem usar submodules. Como o worktree principal já estava sujo, a operação foi feita em uma worktree temporária da branch local `codex/project_subtrees_bootstrap`, e o snapshot resultante foi copiado para `project/external/`.

Isso preserva um caminho claro de sincronização futura sem exigir limpeza imediata da branch `alphaquantum`.

## Perfis de ambiente

- `cpu`: perfil local para macOS/arm64, com JAX CPU e smoke tests.
- `cuda`: perfil oficial para Linux/A100, espelhando os pins públicos do demo.

## Artefatos importantes

- `results/reproducibility/bootstrap_manifest.json`: manifesto único com ambiente, vendors e comandos executados.
- `results/logs/demo/`: logs do smoke/control do demo.
- `results/compile/`: saídas do `compile`.
- `results/resynth/`: saídas do `resynth`.
