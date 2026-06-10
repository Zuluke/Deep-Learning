"""Debug helper: compare two CNOT+diagonal QASM blocks exactly.

Both circuits must consist only of {cx, cz, x, z, s, sdg, t, tdg}. Each maps a
computational basis state |x> to e^{i pi/4 * p(x)} |L(x)>, with L linear-affine
over GF(2) and p(x) an integer mod 8. We enumerate all 2^n inputs, compare the
basis maps, and fit the phase difference as a multilinear polynomial mod 8.
"""

from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

GATE_RE = re.compile(r"^\s*(cx|cz|x|z|s|sdg|t|tdg)\s+q\[(\d+)\](?:\s*,\s*q\[(\d+)\])?\s*;")


def parse(path: Path):
    gates = []
    nq = None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*qreg\s+q\[(\d+)\]", line)
        if m:
            nq = int(m.group(1))
            continue
        m = GATE_RE.match(line)
        if m:
            name, a, b = m.group(1), int(m.group(2)), m.group(3)
            gates.append((name, a, int(b) if b is not None else None))
        elif line.strip() and not line.startswith(("OPENQASM", "include", "//")):
            raise ValueError(f"Unhandled line in {path}: {line!r}")
    return nq, gates


def simulate(nq: int, gates) -> tuple[np.ndarray, np.ndarray]:
    """Return (out_state[x], phase_mod8[x]) for every basis input x."""
    n_states = 1 << nq
    xs = np.arange(n_states, dtype=np.int64)
    state = xs.copy()
    phase = np.zeros(n_states, dtype=np.int64)
    for name, a, b in gates:
        bit_a = (state >> a) & 1
        if name == "cx":
            state ^= bit_a << b
        elif name == "x":
            state ^= 1 << a
        elif name == "cz":
            phase += 4 * (bit_a & ((state >> b) & 1))
        elif name == "z":
            phase += 4 * bit_a
        elif name == "s":
            phase += 2 * bit_a
        elif name == "sdg":
            phase += 6 * bit_a
        elif name == "t":
            phase += bit_a
        elif name == "tdg":
            phase += 7 * bit_a
        else:
            raise ValueError(name)
    return state, phase % 8


def fit_multilinear_mod8(nq: int, values: np.ndarray, max_degree: int = 4):
    """Mobius transform: coefficients of the multilinear polynomial mod 8."""
    coeffs = {}
    residual = values.astype(np.int64).copy()
    for degree in range(0, max_degree + 1):
        for subset in combinations(range(nq), degree):
            mask = 0
            for q in subset:
                mask |= 1 << q
            x = mask
            c = residual[x] % 8
            if c:
                coeffs[subset] = int(c)
                sel = (np.arange(len(values)) & mask) == mask
                residual[sel] -= c
    residual %= 8
    return coeffs, int(np.count_nonzero(residual))


def main() -> int:
    path_a, path_b = Path(sys.argv[1]), Path(sys.argv[2])
    nq_a, gates_a = parse(path_a)
    nq_b, gates_b = parse(path_b)
    nq = max(nq_a, nq_b)
    state_a, phase_a = simulate(nq, gates_a)
    state_b, phase_b = simulate(nq, gates_b)

    if np.array_equal(state_a, state_b):
        print("linear parts: EQUAL")
    else:
        diff = state_a ^ state_b
        print("linear parts: DIFFER")
        print("  xor-diff distinct values:", sorted(set(diff.tolist()))[:10])
        # affine offset and linear defect rows
        offset = int(diff[0])
        print(f"  affine offset (diff at x=0): {offset}")
        lin = diff ^ offset
        basis_rows = {q: int(lin[1 << q]) for q in range(nq) if lin[1 << q]}
        print(f"  linear defect on basis vectors: {basis_rows}")

    dphase = (phase_a - phase_b) % 8
    if not dphase.any():
        print("phase difference: ZERO")
        return 0
    coeffs, resid = fit_multilinear_mod8(nq, dphase)
    print(f"phase difference (a - b) mod 8, multilinear coefficients (residual nonzeros beyond deg4: {resid}):")
    for subset in sorted(coeffs, key=lambda s: (len(s), s)):
        print(f"  {subset}: {coeffs[subset]}")
    max_deg = max(len(s) for s in coeffs)
    print(f"max degree: {max_deg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
