import sys
from pathlib import Path

import sympy as sp

# make sure we can import your modules
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from encoding_utils import linear_system, rule_to_row
from simplify import RewriteSimplifier
from rules import RuleSet


def compute_pivot_basis(equations, symbols):
    """
    Replicate the demo's pivot‐trace elimination:
      - build the rewrite system
      - for each rule in order, attempt row_reduce against the current basis
      - if non‐redundant, eliminate older basis rules against it, then add it
    """
    rs = linear_system(equations, symbols)
    simplifier = RewriteSimplifier(rs)
    basis = RuleSet(rewrite_func=rs.rewrite_func)  # empty basis
    
    for rule in rs.rules:
        red = simplifier.row_reduce(basis, rule, max_steps=20)
        if red is None:
            continue
        # eliminate old basis rules against the new one
        temp = RuleSet([red], rewrite_func=rs.rewrite_func)
        to_replace = []
        for old in basis.rules:
            nl = temp.normal_form(old.lhs, max_steps=20)
            nr = temp.normal_form(old.rhs, max_steps=20)
            if (nl, nr) != (old.lhs, old.rhs):
                to_replace.append((old, nl, nr))
        for old, nl, nr in to_replace:
            basis.rules.remove(old)
            if nl != nr:
                basis.add(nl, nr)
        # finally add the new pivot rule
        basis.add(red.lhs, red.rhs)

    return basis


def assert_rref_equivalence(equations, symbols):
    # get the pivot basis
    basis = compute_pivot_basis(equations, symbols)
    # convert basis rules to matrix rows
    collapse_rows = [rule_to_row(r, symbols) for r in basis.rules]
    collapse_mat = sp.Matrix(collapse_rows)

    # build and rref the original augmented system
    aug = []
    for eq in equations:
        e = eq.lhs - eq.rhs
        coeffs = [e.coeff(s, 1) for s in symbols]
        const = -e.subs({s: 0 for s in symbols})
        aug.append(coeffs + [const])
    full = sp.Matrix(aug)
    rref_full, _ = full.rref()
    # drop all zero rows
    target_rows = [list(rref_full.row(i)) for i in range(rref_full.rows) if any(rref_full.row(i))]
    target = sp.Matrix(target_rows)

    # now compare
    got = collapse_mat.rref()[0]
    assert got == target, f"\nGot:\n{got}\nExpected:\n{target}"


def test_gaussian_elimination_equivalence():
    x, y = sp.symbols("x y")
    eqs = [
        sp.Eq(2 * x + 3 * y, 7),
        sp.Eq(4 * x + 6 * y, 14),
        sp.Eq(6 * x - 3 * y, 9),
    ]
    assert_rref_equivalence(eqs, (x, y))


def test_single_equation():
    x = sp.symbols("x")
    eqs = [sp.Eq(x, 5)]
    assert_rref_equivalence(eqs, (x,))


def test_3x3_full_rank():
    x, y, z = sp.symbols("x y z")
    eqs = [
        sp.Eq(x + 2*y - z, 1),
        sp.Eq(2*x - y + 3*z, 4),
        sp.Eq(-x + y + 2*z, -1),
    ]
    assert_rref_equivalence(eqs, (x, y, z))


def test_underdetermined_system():
    x, y = sp.symbols("x y")
    eqs = [
        sp.Eq(x + y, 2),
        sp.Eq(2*x + 2*y, 4),
    ]
    # for underdetermined, we should get exactly one pivot row
    basis = compute_pivot_basis(eqs, (x, y))
    rows = [rule_to_row(r, (x, y)) for r in basis.rules]
    mat = sp.Matrix(rows)
    # compare to the single nonzero rref row [1,1,2]
    expected = sp.Matrix([[1, 1, 2]])
    assert mat.rref()[0] == expected


def test_inconsistent_system():
    x, y = sp.symbols("x y")
    eqs = [
        sp.Eq(x + y, 1),
        sp.Eq(x + y, 2),
    ]
    basis = compute_pivot_basis(eqs, (x, y))
    rows = [rule_to_row(r, (x, y)) for r in basis.rules]
    # expect at least one inconsistent row [0,0,c] with c != 0
    assert any(r[0] == 0 and r[1] == 0 and r[2] != 0 for r in rows), f"No inconsistency in {rows}"
