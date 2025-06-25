# Rewrite Normalization
> Structural collapse, entropy, and canonical rewriting in Python + Lean 4

![MIT licence](https://img.shields.io/badge/licence-MIT-blue)

This repository contains:
* **Python reference implementation** of rewrite-normalization metrics  
  (collapse map, structural entropy, Gaussian-elimination demo, etc.).
* **Lean 4 formalisation** of key theorems (termination, monotonicity).
* A growing **benchmark suite** for algebraic and combinatorial systems.

---

## Overview

Many algebraic procedures (Gaussian elimination, Gröbner bases, term rewriting)  
can be reframed as *eliminating redundant rules* under a quantitative metric.

| Metric | Intuition | Formula |
|--------|-----------|---------|
| **Collapse** `E↓` | How much the system compresses expressions | `∑ᵢ log |fiberᵢ|` |
| **Entropy** `H`   | Distributional “uncertainty’’ among fibers | `–∑ᵢ pᵢ log pᵢ` |

Experiments show our collapse metric reproduces pivot choices in Gaussian
elimination and detects a phase-shift in `Sₙ` at the solvability threshold.

---

## Python Quick Start

```bash
git clone https://github.com/rewriteunity/rewrite-normalization.git
cd rewrite-normalization

# 1.  Install runtime deps
pip install -e .

# 2.  Run the interactive demo
python demos.py

# 3.  Execute the test suite
pytest
```
## Lean Quick Start
```bash
cd lean_proof

# (one-time) pull Mathlib & other deps
lake update

# build all Lean modules
lake build
```
