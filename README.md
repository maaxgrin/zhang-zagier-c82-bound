# C82 Zhang--Zagier bound

This repository contains Python/Sage-friendly code for an upper bound for the
Zhang--Zagier height constant in the optimization-problems entry

https://teorth.github.io/optimizationproblems/constants/82a.html

It improves the recorded bound

```text
C82 <= 0.25444
```

to the certified bound

```text
C82 <= 0.2536331090204145.
```

The construction uses the polynomials `P1,...,P9,Q1,Q2` introduced by Doche
[Doc01b].  Starting from the denominator factors `Q1,Q2`, the code adds the two
perturbative factors

```text
R0 = Q1 - P1^5 P2^5 P4 P8,
R2 = Q2 + P1^5 P2^5 P4 P7.
```

The denominator used for the final certificate is

```text
Q = Q1 * Q2 * R0 * R2 * P7 * P9.
```

For this `Q`, optimizing the perturbative numerator exponents gives the
numerical value

```text
0.253626920959.
```

The certified statement is the slightly larger interval upper bound

```text
0.2536331090204145.
```

## Files

- `zzopt/`: search, refinement, and interval-certification code.
- `tests/`: regression tests, including reproduction of the previous baseline.
- `runs/qstar_cert_roots_262144_target253635.jsonl`: certified run for the
  displayed denominator.
- `runs/qstar_targeted_refined.jsonl`: numerical refinement for the displayed
  denominator.
- `runs/exhaustive_q1q2_squarefree_grid1024_report.md`: finite-family search
  report with `Q1,Q2` required and `R0,R2,P1,...,P9` optional.

## Verification

Install the package in editable mode and run the tests:

```bash
python -m pip install -e .
python -m pytest -q
```

The included certification run uses Arb via `python-flint`, certified root
enclosures for the factors of `Q(chi(t))`, and a panelwise upper bound for the
log-plus integral.  The run has

```text
certified = True
failed_panels = 0
```

The code also reproduces Doche's baseline construction from [Doc01b] as

```text
0.2544361224.
```

As a finite search check, the code exhausts square-free products with `Q1,Q2`
required and optional factors `R0,R2,P1,...,P9`, subject to `deg(Q) <= 220`.
This checks 2048 products and ranks the displayed denominator first on the LP
screening grid.

## Reference

[Doc01b] C. Doche, "Minorations de hauteurs et petits degres", Journal de
Theorie des Nombres de Bordeaux 13 (2001), no. 1, 103--110.
