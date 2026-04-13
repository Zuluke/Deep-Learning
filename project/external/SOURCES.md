# Vendor Sources

Este diretório contém snapshots vendorizados sem patches locais nesta fase.

## Estratégia usada

- método: `git subtree` em uma worktree temporária;
- branch auxiliar local: `codex/project_subtrees_bootstrap`;
- data de importação: `2026-04-13`;
- motivo da branch auxiliar: o worktree principal já estava sujo, então o import precisou ser isolado para não tocar nas mudanças existentes.

## Repositórios fixados

### `alphatensor_quantum`

- upstream: `https://github.com/google-deepmind/alphatensor_quantum.git`
- ref: `main`
- commit fixado: `3def81a2a42666416a4a8041eea6e1bc98bc8e9f`
- vendor path: `project/external/alphatensor_quantum`

Sync futuro:

```bash
git subtree pull --prefix=project/external/alphatensor_quantum https://github.com/google-deepmind/alphatensor_quantum.git main --squash
```

### `circuit-to-tensor`

- upstream: `https://github.com/tlaakkonen/circuit-to-tensor.git`
- ref: `main`
- commit fixado: `4d6f1f9bc2f3cbad674cb8ebef7544af6853545d`
- vendor path: `project/external/circuit-to-tensor`

Sync futuro:

```bash
git subtree pull --prefix=project/external/circuit-to-tensor https://github.com/tlaakkonen/circuit-to-tensor.git main --squash
```

## Observação importante

Os snapshots atuais em `project/external/` são equivalentes ao conteúdo importado pela branch auxiliar. Quando a branch principal estiver limpa, essa branch local pode ser usada como referência para incorporar o histórico `subtree` ao fluxo normal do repositório.
