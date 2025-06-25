from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple
import math
from collections import deque

from rules import RuleSet

# Enable verbose debugging for entropy and collapse calculations
VERBOSE = False

@dataclass
class TermGraph:
    """Directed graph capturing rewrite expansions."""
    children: Dict[Any, Set[Any]] = field(default_factory=dict)
    parents: Dict[Any, Set[Any]] = field(default_factory=dict)

    def add_edge(self, src: Any, dst: Any) -> None:
        self.children.setdefault(src, set()).add(dst)
        self.parents.setdefault(dst, set()).add(src)

@dataclass
class CollapseMapIncremental:
    """Incrementally track collapse‑fibers during exploration."""
    term_to_collapse: Dict[Any, Any] = field(default_factory=dict)
    collapse_fibers: Dict[Any, Set[Any]] = field(default_factory=dict)

    def register(self, term: Any, collapse: Any) -> None:
        self.term_to_collapse[term] = collapse
        self.collapse_fibers.setdefault(collapse, set()).add(term)


_global_ruleset: RuleSet | None = None

def _init_pool(ruleset: RuleSet) -> None:
    global _global_ruleset
    _global_ruleset = ruleset

def generator_terms(ruleset: RuleSet) -> Set[Any]:
    """Return a set of all LHS and RHS terms in the ruleset."""
    gens: Set[Any] = set()
    for r in ruleset.rules:
        gens.add(r.lhs)
        gens.add(r.rhs)
    return gens

def _expand_term(term: Any) -> Tuple[Any, Set[Any]]:
    """Pool worker: apply every rule once to a term."""
    assert _global_ruleset is not None
    res: Set[Any] = set()
    for rule in _global_ruleset.rules:
        try:
            cand = _global_ruleset.rewrite_func(term, rule)
        except Exception:
            continue
        if cand != term:
            res.add(cand)
    return term, res


def compute_collapse(
    ruleset: RuleSet,
    max_depth: int = 3,
    normalize_fn: Callable[[Any], Any] | None = None,
    processes: int | None = None,
    combine: Callable[[Any, Any], Any] | Iterable[Callable[[Any, Any], Any]] | None = None,
    generators: Iterable[Any] | None = None,
) -> Iterable[Dict[str, Any]]:
    """
    Explore the rewrite‑space up to `max_depth`.  At each depth yield a record:

      {
        'depth': int,
        'rules':   current number of rules,
        'operators': number of combine ops,
        'terms':   total distinct terms seen,
        'collapse_forms': distinct normals,
        'E_down': sum(log fiber_size),
        'entropy': Shannon entropy in bits,
        'collapse_ratio': terms / collapse_forms,
        'avg_fiber': average fiber size
      }
    """
    if normalize_fn is None:
        normalize_fn = ruleset.normal_form

    # decide how many combine‑operators we have
    if combine is None:
        combine_ops: List[Any] = []
    elif callable(combine):
        combine_ops = [combine]
    else:
        combine_ops = list(combine)

    seen: Set[Any] = set()
    
    # Allow custom generators to select fibers for computation.
    if generators is None:
        generators = generator_terms(ruleset)
    next_terms = set(generators)

    cmap = CollapseMapIncremental()
    graph = TermGraph()

    with Pool(processes=processes, initializer=_init_pool, initargs=(ruleset,)) as pool:
        for depth in range(max_depth):
            current = next_terms
            next_terms = set()

            # register normals for new terms
            for t in current:
                if t not in seen:
                    seen.add(t)
                    norm = normalize_fn(t)
                    cmap.register(t, norm)

            # if no expansion, still yield the record
            if not current:
                yield _build_record(depth, seen, cmap, ruleset, combine_ops)
                continue

            # parallel expansion
            results = pool.map(_expand_term, list(current))
            for t, expanded in results:
                for t2 in expanded:
                    graph.add_edge(t, t2)
                    if t2 not in seen:
                        next_terms.add(t2)

            yield _build_record(depth, seen, cmap, ruleset, combine_ops)


def _build_record(
    depth: int,
    seen: Set[Any],
    cmap: CollapseMapIncremental,
    ruleset: RuleSet,
    combine_ops: List[Any],
) -> Dict[str, Any]:
    fiber_sizes = [len(f) for f in cmap.collapse_fibers.values()]
    total_terms = len(seen)
    num_rules = len(ruleset.rules)
    num_ops   = len(combine_ops)
    num_forms = len(cmap.collapse_fibers)

    if fiber_sizes:
        if VERBOSE:
            print(f"[DEBUG] Computing record at depth {depth}")
            print(f"[DEBUG] Fiber sizes: {fiber_sizes}")
        collapse = sum(math.log(sz) for sz in fiber_sizes)
        total = sum(fiber_sizes)
        # compute entropy in bits
        entropy = 0.0
        for sz in fiber_sizes:
            p = sz / total
            contrib = -p * math.log2(p)
            if VERBOSE:
                print(f"[DEBUG] p={p:.6f}, contrib to entropy={contrib:.6f}")
            entropy += contrib
        if VERBOSE:
            print(f"[DEBUG] Sum log fiber sizes (E_down) = {collapse:.6f}")
            print(f"[DEBUG] Computed entropy      = {entropy:.6f}")
        avg_fiber = total / num_forms
        ratio    = total_terms / num_forms
    else:
        collapse = entropy = avg_fiber = ratio = 0.0

    return {
        "depth": depth,
        "rules": num_rules,
        "operators": num_ops,
        "terms": total_terms,
        "collapse_forms": num_forms,
        "E_down": collapse,
        "entropy": entropy,
        "avg_fiber": avg_fiber,
        "collapse_ratio": ratio,
    }


def get_collapse_endpoint(
    ruleset: RuleSet,
    max_depth: int = 3,
    normalize_fn: Callable[[Any], Any] | None = None,
    processes: int | None = None,
    combine: Callable[[Any, Any], Any] | Iterable[Callable[[Any, Any], Any]] | None = None,
    generators: Iterable[Any] | None = None,
) -> Dict[str, Any]:
    """
    Run compute_collapse and return the *last* record dict.
    """
    last = deque(
        compute_collapse(ruleset,
                         max_depth=max_depth,
                         normalize_fn=normalize_fn,
                         processes=processes,
                         combine=combine,
                         generators=generators),
        maxlen=1
    )
    if not last:
        # empty exploration
        return {"depth": 0, "rules": len(ruleset.rules), "operators": 0,
                "terms": 0, "collapse_forms": 0,
                "E_down": 0.0, "entropy": 0.0,
                "avg_fiber": 0.0, "collapse_ratio": 0.0}
    return last[0]

__all__ = [
    "generator_terms",
    "compute_collapse",
    "get_collapse_endpoint",
]
