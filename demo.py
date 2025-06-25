#!/usr/bin/env python3
"""
demo_all.py

Unified demo script illustrating all core empirical cases:
  1. Gaussian elimination as collapse rewriting
  2. Entropy measurement on a probabilistic rewrite system
  3. Toy phase-tracking visualization
  4. Compression exponents for symmetric groups Sₙ
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from simplify import RewriteSimplifier
from rewrite_metrics import compute_collapse, get_collapse_endpoint
from rules import RuleSet
from encoding_utils import linear_system, probabilistic_system, symmetric_phase_shift


def demo_gaussian_elimination(out_dir: Path) -> None:
    """
    Show step-by-step rewrite collapse ↔ Gaussian elimination by
    printing a textbook-style pivot and elimination trace, then
    reporting the final collapse/entropy profile.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build a random 4×4 integer system Ax = b
    rng = np.random.RandomState(0)
    A = rng.randint(-5, 6, size=(4, 4))
    b = rng.randint(-5, 6, size=(4, 1))
    x = sp.symbols("x0:4")
    equations = [
        sp.Eq(sum(int(A[i, j]) * x[j] for j in range(4)), int(b[i, 0]))
        for i in range(4)
    ]

    # Sort so we pivot in order x0, x1, …
    def pivot_key(eq):
        expr = eq.lhs - eq.rhs
        for idx, var in enumerate(x):
            if expr.coeff(var, 1) != 0:
                return idx
        return len(x)
    equations.sort(key=pivot_key)

    print("\nInitial system:")
    for i, eq in enumerate(equations, start=1):
        print(f" Row {i}: {eq}")

    # 2) Build rewrite system & simplifier
    rs = linear_system(equations, x)
    simplifier = RewriteSimplifier(rs)

    # 3) Perform pivot-style elimination
    basis = RuleSet(rewrite_func=rs.rewrite_func)
    for step, rule in enumerate(rs.rules, start=1):
        red = simplifier.row_reduce(basis, rule, max_steps=20)
        if red is None:
            print(f"\nStep {step}: {rule.lhs}->{rule.rhs} redundant, skipped")
            continue

        print(f"\nStep {step}: Pivot on {red.lhs} → {red.rhs}")
        # eliminate against existing basis
        temp = RuleSet([red], rewrite_func=rs.rewrite_func)
        for other in list(basis.rules):
            nl = temp.normal_form(other.lhs, max_steps=20)
            nr = temp.normal_form(other.rhs, max_steps=20)
            if (nl, nr) != (other.lhs, other.rhs):
                print(f"  Eliminate {other.lhs}->{other.rhs} → {nl}->{nr}")
                basis.rules.remove(other)
                if nl != nr:
                    basis.add(nl, nr)
        basis.add(red.lhs, red.rhs)

        print("  Current basis:")
        for r in basis.rules:
            print("   ", f"{r.lhs} -> {r.rhs}")

    # 4) Print final collapse & entropy profile
    profile = get_collapse_endpoint(basis, max_depth=4)
    print("\nFinal Gaussian-elim collapse profile:")
    for key in ("depth", "rules", "operators", "terms", "collapse_forms", "E_down", "entropy"):
        print(f"  {key:15s}: {profile[key]}")


def demo_entropy_probabilistic(max_depth: int = 1) -> None:
    """
    Show that Shannon entropy via collapse matches the direct formula.
    """
    counts = {"a": 3, "b": 1, "c": 2}
    ruleset, gens = probabilistic_system(counts)

    records = list(compute_collapse(ruleset, max_depth=2, generators=gens))
    H = records[0]["entropy"]

    # direct combinatorial entropy
    total = sum(counts.values())
    direct = -sum((c/total) * math.log2(c/total) for c in counts.values())

    print("\nEntropy derivation demo:")
    print(f"Counts: {counts}")
    print(f"Entropy (collapse-based): {H:.6f}")
    print(f"Entropy (direct):          {direct:.6f}")
    print(f"Difference:               {abs(H-direct):.2e}")


def demo_symmetric_phase_shift(
    out_dir: Path,
    degrees: list[int],
    depth: int,
    step: int
) -> None:
    """
    Compute and plot compression exponents α for symmetric groups S_n.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = symmetric_phase_shift(degrees, depth=depth, step=step)
    print("\nSymmetric group phase-shift demo:")
    print(df.to_string(index=False))

    plt.figure()
    plt.plot(df["degree"], df["alpha"], marker="o", label="α")
    plt.axvline(5, color="gray", linestyle="--", label="unsolvable threshold")
    plt.xlabel("Degree n of Sₙ")
    plt.ylabel("Compression exponent α")
    plt.title("Phase shift in symmetric groups")
    plt.legend()
    plt.savefig(out_dir / "symmetric_phase_shift.png")
    plt.clf()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all rewrite demos")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("gauss", help="Gaussian elimination demo")
    ent = sub.add_parser("entropy", help="Entropy derivation demo")
    ent.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Collapse exploration depth for entropy (default: 1)",
    )
    p_shift = sub.add_parser("phase-shift", help="Symmetric group demo")
    p_shift.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=list(range(3, 8)),
        help="Degrees to test (default: 3–7)",
    )
    p_shift.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Exploration depth (default: 6)",
    )
    p_shift.add_argument(
        "--step",
        type=int,
        default=2,
        help="Adjacent‐transposition step (default: 2)",
    )
    p_shift.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Ensure defaults are present even when cmd='all'
    if not hasattr(args, "out"):
        args.out = Path("figures")
    if not hasattr(args, "degrees"):
        args.degrees = list(range(3, 8))
    if not hasattr(args, "depth"):
        args.depth = 6
    if not hasattr(args, "step"):
        args.step = 2

    cmd = args.cmd or "all"
    out = getattr(args, "out", Path("figures"))

    if cmd in ("all", "gauss"):
        demo_gaussian_elimination(out)
    if cmd in ("all", "entropy"):
        # pass depth to entropy demo, defaulting to 1
        demo_entropy_probabilistic(args.depth if hasattr(args, "depth") else 1)
    if cmd in ("all", "phase-shift"):
        demo_symmetric_phase_shift(out, args.degrees, args.depth, args.step)


if __name__ == "__main__":
    main()
