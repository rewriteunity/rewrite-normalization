# tests/test_entropy_derivation.py

import math
import pytest

from encoding_utils import probabilistic_system
from rewrite_metrics import compute_collapse

@pytest.mark.parametrize("counts", [
    {"a": 3, "b": 1, "c": 2},
    {"x": 5, "y": 5, "z": 5},
    {"only": 10},
])
def test_shannon_entropy_consistency(counts):
    """
    Shannon entropy from rewrite‐collapse metrics should equal direct Shannon entropy.
    """
    rs, gens = probabilistic_system(counts)

    # compute_collapse yields one record per depth from 0..1
    records = list(compute_collapse(rs, max_depth=1, generators=gens))
    # depth=0 record has no rewrites applied, so fibers == input counts
    rec = records[0]
    entropy = rec["entropy"]

    total = sum(counts.values())
    expected = -sum((c/total) * math.log2(c/total) for c in counts.values())

    assert entropy == pytest.approx(expected, rel=1e-12)


def test_uniform_distribution_entropy():
    """
    A uniform distribution over N outcomes at depth=0 should have entropy log2(N).
    """
    counts = {"a": 1, "b": 1, "c": 1}
    rs, gens = probabilistic_system(counts)

    records = list(compute_collapse(rs, max_depth=1, generators=gens))
    rec = records[0]
    entropy = rec["entropy"]

    expected = math.log2(len(counts))
    assert entropy == pytest.approx(expected, rel=1e-12)


def test_zero_entropy_for_singleton():
    """
    A single‐outcome system at depth=0 should have zero entropy.
    """
    counts = {"only": 42}
    rs, gens = probabilistic_system(counts)

    records = list(compute_collapse(rs, max_depth=1, generators=gens))
    rec = records[0]
    entropy = rec["entropy"]

    assert entropy == pytest.approx(0.0, abs=1e-15)
