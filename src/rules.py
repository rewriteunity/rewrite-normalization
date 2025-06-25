"""Symbolic rewriting infrastructure used across the structural collapse engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Callable, Any

import sympy as sp


@dataclass(frozen=True)
class Rule:
    """Simple rewrite rule ``lhs`` -> ``rhs``.

    ``lhs`` and ``rhs`` may be any hashable objects. SymPy expressions are the
    default but generic terms such as tuples can be used for more specialised
    rewriting systems.
    """

    lhs: Any
    rhs: Any


class RuleSet:
    """Collection of symbolic rewrite rules with reduction utilities."""

    def __init__(
        self,
        rules: Iterable[Rule] | None = None,
        rewrite_func: Callable[[Any, Rule], Any] | None = None,
    ) -> None:
        self.rules: List[Rule] = list(rules or [])
        self.rewrite_func = rewrite_func or self._default_rewrite

    def add(self, lhs: Any, rhs: Any) -> None:
        """Add a new rule to the set."""
        self.rules.append(Rule(lhs, rhs))

    def _default_rewrite(self, expr: Any, rule: Rule) -> Any:
        if isinstance(expr, sp.Basic):
            return expr.xreplace({rule.lhs: rule.rhs})
        if isinstance(expr, (list, tuple)):
            res = [self._default_rewrite(e, rule) for e in expr]
            return type(expr)(res)
        return rule.rhs if expr == rule.lhs else expr

    def apply_once(self, expr: Any, exclude: Iterable[Rule] | None = None) -> Any:
        """Apply all rules to expr exactly once."""

        out = expr
        skip = set(exclude or [])
        for rule in self.rules:
            if rule in skip:
                continue
            out = self.rewrite_func(out, rule)
        return out

    def normal_form(
        self, expr: Any, max_steps: int = 10, exclude: Iterable[Rule] | None = None
    ) -> Any:
        """Return the normal form of expr under the rules.

        Parameters
        ----------
        expr:
            Expression to normalise.
        max_steps:
            Maximum number of rewrite passes to attempt.
        exclude:
            Optional iterable of rules to omit during rewriting.
        """

        prev = None
        current = expr
        steps = 0
        while current != prev and steps < max_steps:
            prev = current
            current = self.apply_once(current, exclude=exclude)
            steps += 1
        return current

    def reduce_all(self) -> None:
        """Reduce all rules against each other."""
        for i, rule in enumerate(list(self.rules)):
            lhs = self.normal_form(rule.lhs, exclude=[rule])
            rhs = self.normal_form(rule.rhs, exclude=[rule])
            self.rules[i] = Rule(lhs, rhs)

    def analyze_rewrite(self, expr: Any, max_steps: int = 10) -> RewriteAnalysis:
        """Return a RewriteAnalysis describing how expr rewrites.

        Parameters
        ----------
        expr:
            Expression to analyse.
        max_steps:
            Maximum number of rewrite passes to attempt.
        """

        seen = {expr}
        current = expr
        steps = 0
        while steps < max_steps:
            next_expr = current
            for rule in self.rules:
                candidate = self.rewrite_func(current, rule)
                if candidate != current:
                    next_expr = candidate
                    break
            steps += 1
            if next_expr == current:
                return RewriteAnalysis(next_expr, steps, False, len(seen))
            if next_expr in seen:
                seen.add(next_expr)
                return RewriteAnalysis(next_expr, steps, True, len(seen))
            seen.add(next_expr)
            current = next_expr
        return RewriteAnalysis(current, steps, True, len(seen))


@dataclass
class RewriteAnalysis:
    """Summary information about a rewrite sequence."""

    final: Any
    steps: int
    loop: bool
    unique: int

    def explosion_rate(self) -> float:
        """Return the unique expressions seen per rewrite step."""
        return self.unique / self.steps if self.steps else 0.0


def parse_rule_lines(
    lines: Iterable[str], parse_sympy: bool = True
) -> RuleSet:
    """Return a RuleSet parsed from lines."""

    rs = RuleSet()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "->" not in line:
            continue
        lhs_str, rhs_str = [p.strip() for p in line.split("->", 1)]
        if parse_sympy:
            try:
                lhs_expr = sp.sympify(lhs_str)
                rhs_expr = sp.sympify(rhs_str)
            except Exception:
                lhs_expr, rhs_expr = lhs_str, rhs_str
        else:
            lhs_expr, rhs_expr = lhs_str, rhs_str
        rs.add(lhs_expr, rhs_expr)
    return rs
