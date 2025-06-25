from __future__ import annotations

from typing import Callable, Iterable, Tuple, Sequence, Dict, Any, List
from itertools import product
from functools import lru_cache, partial

import sympy as sp
import numpy as np
import pandas as pd

from rules import Rule, RuleSet

############# Polynomial system encoding #########################


def _polynomial_rewrite(
    expr: sp.Basic,
    rule: Rule,
    symbols: tuple[sp.Symbol, ...]
) -> sp.Basic:
    """
    Top-level worker function: reduce `expr` modulo `rule.lhs` (assuming `rule.rhs==0`).
    Must live at module scope so multiprocessing can pickle it.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    if not (isinstance(rule.lhs, sp.Basic) and rule.rhs == 0):
        return expr

    # perform polynomial remainder
    poly_expr = sp.Poly(expr, *symbols)
    poly_rule = sp.Poly(rule.lhs, *symbols)
    rem = poly_expr.rem(poly_rule)
    return rem.as_expr()


def polynomial_rewrite_factory(
    symbols: Iterable[sp.Symbol]
) -> Callable[[sp.Basic, Rule], sp.Basic]:
    """
    Returns a pickleable rewrite function by fixing `symbols` in
    the top‐level `_polynomial_rewrite`.
    """
    #   convert symbols to a tuple so it's hashable/pickleable
    symbols_tuple = tuple(symbols)
    return partial(_polynomial_rewrite, symbols=symbols_tuple)


def polynomial_rules(polys: Iterable[sp.Basic], symbols: Sequence[sp.Symbol]) -> RuleSet:
    """Return a ``RuleSet`` mapping each polynomial to ``0`` using remainder reduction."""

    rs = RuleSet(rewrite_func=polynomial_rewrite_factory(symbols))
    for p in polys:
        rs.add(p, sp.Integer(0))
    return rs


############# Probabilistic system encoding #########################

def _micro_rewriter(expr: Any, rule: Rule) -> Any:
    """
    Top‐level worker function for probabilistic microstate rewrites,
    so that multiprocessing can pickle it.
    """
    return rule.rhs if expr == rule.lhs else expr

def probabilistic_system(counts: Dict[Any, int]) -> Tuple[RuleSet, List[Any]]:
    """Return rules capturing a discrete distribution."""
    rs = RuleSet(rewrite_func=_micro_rewriter)
    micros: List[Any] = []
    for event, n in counts.items():
        for i in range(n):
            micro = (event, i)
            rs.add(micro, event)
            micros.append(micro)
    return rs, micros



################### Symmetry Encoding Utils ######################


def identity(size: int) -> Tuple[int, ...]:
    """The identity permutation in S_size."""
    return tuple(range(size))


def compose(p: Tuple[int, ...], q: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compose two permutations p∘q."""
    return tuple(p[i] for i in q)


def transposition(k: int, size: int, step: int = 1) -> Tuple[int, ...]:
    """
    The adjacent transposition (k, k+step) in S_size.

    Parameters
    ----------
    k : int
        First index of the transposition.
    size : int
        Degree of the symmetric group.
    step : int, default=1
        Distance between swapped elements.
    """
    j = (k + step) % size
    p = list(range(size))
    p[k], p[j] = p[j], p[k]
    return tuple(p)


@lru_cache(maxsize=None)
def word_to_perm(
        word: Tuple[int, ...], size: int, step: int = 1
) -> Tuple[int, ...]:
    """
    Convert a word over adjacent-transposition generators into a permutation.

    word[i] ∈ {0, …, size–2} picks the transposition at that position.
    """
    perm = identity(size)
    for g in word:
        perm = compose(transposition(g, size, step), perm)
    return perm


def symmetric_ratios(
        degree: int, max_depth: int, step: int = 1
) -> pd.DataFrame:
    """
    Compute raw vs. normal counts and compression ratios for S_degree.

    Parameters
    ----------
    degree : int
        Degree of the symmetric group.
    max_depth : int
        Maximum word length to enumerate.
    step : int, default=1
        Adjacent-transposition distance.

    Returns
    -------
    DataFrame with columns n, raw, normal, ratio
    """
    k = degree - 1
    data = []
    for n in range(1, max_depth + 1):
        raw = k ** n
        # enumerate all words of length n over k generators
        normals = {
            word_to_perm(word, degree, step)
            for word in product(range(k), repeat=n)
        }
        data.append({
            "n": n,
            "raw": raw,
            "normal": len(normals),
            "ratio": raw / len(normals),
        })
    return pd.DataFrame(data)


def fit_exponent(df: pd.DataFrame, base: int) -> float:
    """
    Fit compression exponent alpha via log(ratio) ~ alpha * n * log(base).

    Uses a linear fit on n >= 2.
    """
    vals = df[df["n"] >= 2]
    if vals.empty:
        return float("nan")
    x = vals["n"].to_numpy()
    y = np.log(vals["ratio"].to_numpy()) / np.log(base)
    slope, _ = np.polyfit(x, y, 1)
    return slope


def is_solvable_symmetric(degree: int) -> bool:
    """Return True iff S_degree is solvable (degree ≤ 4)."""
    return degree <= 4


def symmetric_phase_shift(
        degrees: Iterable[int], depth: int, step: int = 1
) -> pd.DataFrame:
    """
    Compute compression exponent alpha and solvability for each degree in degrees.

    Returns a DataFrame with columns degree, alpha, solvable.
    """
    records = []
    for d in degrees:
        df = symmetric_ratios(d, depth, step)
        alpha = fit_exponent(df, d - 1)
        records.append({
            "degree": d,
            "alpha": alpha,
            "solvable": is_solvable_symmetric(d),
        })
    return pd.DataFrame(records)


######### Linear System Encoding #########


def linear_system(equations: Iterable[sp.Eq], symbols: Sequence[sp.Symbol]) -> RuleSet:
    """Return rewrite rules encoding a system of linear equations."""
    rs = RuleSet(rewrite_func=polynomial_rewrite_factory(symbols))
    for eq in equations:
        if not isinstance(eq, sp.Equality):
            raise TypeError("equations must be SymPy Eq instances")
        rs.add(sp.expand(eq.lhs - eq.rhs), sp.Integer(0))
    return rs

def rule_to_row(rule: Rule, symbols: Sequence[sp.Symbol]) -> List[sp.Expr]:
    """Return the matrix row corresponding to rule."""
    poly = sp.expand(rule.lhs - rule.rhs)
    p = sp.Poly(poly, *symbols)
    coeffs = [p.coeff_monomial(s) for s in symbols]
    const = -p.subs({s: 0 for s in symbols})
    return coeffs + [const]


def rewrite_matrix(ruleset: RuleSet, symbols: Sequence[sp.Symbol]) -> sp.Matrix:
    """Return the augmented matrix for ruleset"""
    rows = [rule_to_row(r, symbols) for r in ruleset.rules]
    return sp.Matrix(rows)


__all__ = ["rewrite_matrix", "linear_system", "rule_to_row"]
