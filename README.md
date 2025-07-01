# Rewrite Normalization  
> Structural collapse, entropy, and canonical rewriting in Python + Lean 4

![MIT licence](https://img.shields.io/badge/licence-MIT-blue)

This repository contains:

- **Python reference implementation** of rewrite-normalization metrics  
  (collapse map, structural entropy, Gaussian-elimination demo, etc.).
- **Lean 4 formalisation** of key theorems (termination, monotonicity).
- A growing **benchmark suite** for algebraic and combinatorial systems.
- **Emergence demos** (in Jupyter) demonstrating our new “ridge-tracing” framework—see below.

---

## Overview

Many algebraic procedures (Gaussian elimination, Gröbner bases, term rewriting)  
can be reframed as *eliminating redundant rules* under a quantitative metric.

| Metric            | Intuition                                 | Formula                  |
|-------------------|-------------------------------------------|--------------------------|
| **Collapse** `E↓` | How much the system compresses expressions | `∑ₐ log |fiberₐ|`        |
| **Entropy** `H`   | Distributional “uncertainty” among fibers  | `–∑ₐ pₐ log pₐ`          |

Experiments show our collapse metric reproduces pivot choices in Gaussian  
elimination and detects a phase-shift in `Sₙ` at the solvability threshold.

---

## Emergence-Ridge Demos

We’ve recently added a set of interactive Jupyter notebooks exploring the  
“Emergence Gap” and ridge-tracing algorithms on classical examples (dice-sums,  
Rule 110 CA, logistic map, etc.).  This is still exploratory (“less fleshed out”),  
but you can browse and run them here:

- **`EmergenceDemos.ipynb`** — step through the entropy vs. collapse metrics,  
  plot curvature spikes, and inspect critical depths for several systems.

---

## Python Quick Start

```bash
git clone https://github.com/rewriteunity/rewrite-normalization.git
cd rewrite-normalization

# 1. Install runtime deps
pip install -e .

# 2. Run the classic demos
python demos.py

# 3. Launch the Emergence-Ridge notebook
jupyter lab EmergenceDemos.ipynb

# 4. Execute the test suite
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
