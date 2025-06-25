from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Dict, Set, List, Any, Callable
import math

import sympy as sp
import networkx as nx

from rewrite_metrics import get_collapse_endpoint
from typing import TYPE_CHECKING


from rules import Rule, RuleSet


def expr_complexity(expr: Any) -> int:
    """Return a simple complexity measure for ordering terms."""

    if isinstance(expr, sp.Basic):
        return sp.count_ops(expr, visual=False) + len(expr.args)
    if isinstance(expr, (list, tuple)):
        return sum(expr_complexity(e) for e in expr) + len(expr)
    return 1


def orient(lhs: Any, rhs: Any, key: Callable[[Any], Any] = expr_complexity) -> tuple[Any, Any]:
    """
    Return (lhs, rhs) oriented from complex to simple, preserving original order on exact ties.

    Parameters
    ----------
    lhs, rhs:
        Terms to orient.
    key:
        Complexity measure used for ordering. Higher values correspond to more complex terms.
    """

    l_comp = key(lhs)
    r_comp = key(rhs)
    # If one side is strictly more complex, put it first
    if l_comp > r_comp:
        return lhs, rhs
    if r_comp > l_comp:
        return rhs, lhs

    # Complexity ties; try sympy sort keys
    try:
        l_key = sp.default_sort_key(lhs)
        r_key = sp.default_sort_key(rhs)
        # If sort keys are identical, keep original order
        if l_key == r_key:
            return lhs, rhs
        return (lhs, rhs) if l_key > r_key else (rhs, lhs)
    except Exception:
        # Fall back to string comparison
        lhs_str = str(lhs)
        rhs_str = str(rhs)
        if lhs_str == rhs_str:
            return lhs, rhs
        return (lhs, rhs) if lhs_str > rhs_str else (rhs, lhs)


@dataclass
class RewriteSimplifier:
    """Perform rule elimination and produce a collapse rule basis."""

    ruleset: RuleSet
    order_key: Callable[[Any], Any] = expr_complexity
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def row_reduce(self, basis: RuleSet, rule: Rule, max_steps: int) -> Rule | None:
        """Reduce rule against basis and orient it.

        Parameters
        ----------
        basis:
            Current collapse basis used for reduction.
        rule:
            Rule being considered.

        Returns
        -------
        Rule | None
            Rule oriented from complex to simple if it contributes new
            information, otherwise None.
        """

        lhs = basis.normal_form(rule.lhs, max_steps=max_steps)
        rhs = basis.normal_form(rule.rhs, max_steps=max_steps)
        if lhs == rhs:
            return None

        lhs, rhs = orient(lhs, rhs, key=self.order_key)

        lhs = basis.normal_form(lhs, max_steps=max_steps)
        rhs = basis.normal_form(rhs, max_steps=max_steps)
        if lhs == rhs:
            return None

        # If both sides are symbolic expressions try to detect trivial
        # polynomial relations.  When the difference simplifies to 0
        # under SymPy we treat the rule as redundant.
        if isinstance(lhs, sp.Basic) and isinstance(rhs, sp.Basic):
            try:
                if sp.simplify(lhs - rhs) == 0:
                    return None
            except Exception:
                pass

        return Rule(lhs, rhs)

    def collapse(
        self,
        max_nf_steps: int = 10,
        depth: int = 1,
        combine: Any = None,
        alpha: float = 1.0,
        balanced: bool = False,   # you can remove this arg if you never use it
        sort_fn: Callable[[Rule], Any] | None = None,
    ) -> RuleSet:
        """
        Greedy elimination purely on E_down (collapse metric), with optional custom sort order.

        Parameters
        ----------
        max_nf_steps:
            Max steps for normal form reduction.
        depth:
            Depth for collapse metric calculation.
        combine:
            Combine operation(s) for collapse metric.
        alpha:
            Weighting for metrics (unused).
        balanced:
            Placeholder for future use (unused).
        sort_fn:
            Optional function to sort rules before each elimination pass.
            If None, rules are processed in their current order.
        """
        working = RuleSet(self.ruleset.rules, rewrite_func=self.ruleset.rewrite_func)

        # Fully canonicalize by repeated E_down-only pruning
        changed = True
        while changed:
            changed = False
            # Determine processing order
            rule_list: List[Rule] = list(working.rules)
            if sort_fn is not None:
                rule_list = sorted(rule_list, key=sort_fn)

            for rule in rule_list:
                # compute current collapse metric
                E_down = get_collapse_endpoint(
                    ruleset=working,
                    max_depth=depth,
                    normalize_fn=None,
                    processes=None,
                    combine=combine,
                )[
                    "E_down"
                ]
                if E_down <= 0:
                    working.rules.remove(rule)
                    changed = True
                    break
        return working
