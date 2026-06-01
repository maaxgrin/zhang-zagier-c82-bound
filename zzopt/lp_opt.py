"""Discretized linear programming for numerator exponents."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

from .integral import chi_grid, integral_logplus, log_abs_poly_on_chi, log_abs_product_on_chi
from .mahler import log_mahler_product
from .polys import Poly, degree


@dataclass
class LPResult:
    success: bool
    message: str
    q: list[float]
    q_rational: list[str]
    degA: float
    degA_rational: str
    integral: float
    bound_log: float
    degQ: int
    logM: float


def rationalize_q(
    q: Sequence[float],
    numerator_factors: Sequence[Poly],
    degQ: int,
    *,
    max_den: int = 100000,
) -> list[Fraction]:
    fracs = [Fraction(max(0.0, float(x))).limit_denominator(max_den) for x in q]
    deg_a = sum(Fraction(degree(poly), 1) * qi for poly, qi in zip(numerator_factors, fracs))
    if deg_a == 0:
        return fracs
    if deg_a >= degQ:
        scale = Fraction(degQ * max_den - 1, max_den) / deg_a
        fracs = [qi * scale for qi in fracs]
    return fracs


def fraction_list_to_float(q: Sequence[Fraction]) -> list[float]:
    return [float(x) for x in q]


def optimize_exponents_lp(
    Q_factors: Sequence[tuple[Poly, int]],
    numerator_factors: Sequence[Poly],
    *,
    grid_size: int = 8192,
    slack: float = 1e-7,
    max_den: int = 100000,
    logM: float | None = None,
    method: str = "highs",
) -> LPResult:
    grid = chi_grid(grid_size)
    m = len(numerator_factors)
    n = len(grid)
    degQ = sum(degree(poly) * int(mult) for poly, mult in Q_factors)
    if logM is None:
        logM = float(log_mahler_product(Q_factors, fast=True))

    if m == 0:
        integral = integral_logplus(Q_factors, [], [], grid_size, grid=grid)
        return LPResult(
            success=True,
            message="no numerator factors",
            q=[],
            q_rational=[],
            degA=0.0,
            degA_rational="0",
            integral=integral,
            bound_log=(logM + integral) / degQ,
            degQ=degQ,
            logM=logM,
        )

    a_cols = [log_abs_poly_on_chi(poly, grid) for poly in numerator_factors]
    a_mat = np.vstack(a_cols).T
    b_q = log_abs_product_on_chi(Q_factors, grid)

    rows = []
    cols = []
    data = []
    for k in range(n):
        for i in range(m):
            val = a_mat[k, i]
            if val:
                rows.append(k)
                cols.append(i)
                data.append(float(val))
        rows.append(k)
        cols.append(m + k)
        data.append(-1.0)

    deg_row = n
    for i, poly in enumerate(numerator_factors):
        rows.append(deg_row)
        cols.append(i)
        data.append(float(degree(poly)))

    A_ub = sparse.coo_matrix((data, (rows, cols)), shape=(n + 1, m + n)).tocsr()
    b_ub = np.concatenate([b_q, np.array([degQ - slack], dtype=np.float64)])
    c = np.concatenate([np.zeros(m), np.full(n, 1.0 / n)])
    bounds = [(0.0, None)] * (m + n)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    if not res.success:
        return LPResult(
            success=False,
            message=res.message,
            q=[0.0] * m,
            q_rational=["0"] * m,
            degA=0.0,
            degA_rational="0",
            integral=float("inf"),
            bound_log=float("inf"),
            degQ=degQ,
            logM=logM,
        )

    q_float = [max(0.0, float(x)) for x in res.x[:m]]
    q_frac = rationalize_q(q_float, numerator_factors, degQ, max_den=max_den)
    q_eval = fraction_list_to_float(q_frac)
    integral = integral_logplus(Q_factors, numerator_factors, q_eval, grid_size, grid=grid)
    degA_frac = sum(Fraction(degree(poly), 1) * qi for poly, qi in zip(numerator_factors, q_frac))
    return LPResult(
        success=True,
        message=res.message,
        q=q_eval,
        q_rational=[str(x) for x in q_frac],
        degA=float(degA_frac),
        degA_rational=str(degA_frac),
        integral=integral,
        bound_log=(float(logM) + integral) / degQ,
        degQ=degQ,
        logM=float(logM),
    )
