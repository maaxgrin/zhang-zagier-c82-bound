"""Mahler measure computations for Q(z(1-z))."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import mpmath as mp
import numpy as np
import sympy as sp

from .polys import Poly, coeffs_high_to_low, degree, leading_coeff, to_poly


def _quadratic_contribution_complex(xi: complex) -> float:
    disc = 1.0 - 4.0 * xi
    root = np.sqrt(np.complex128(disc))
    r_plus = (1.0 + root) / 2.0
    r_minus = (1.0 - root) / 2.0
    return math.log(max(1.0, abs(r_plus))) + math.log(max(1.0, abs(r_minus)))


def log_mahler_fast(poly: Poly) -> float:
    """Fast double-precision approximation of log M(Q(z(1-z)))."""

    p = to_poly(poly)
    coeffs = np.array(coeffs_high_to_low(p), dtype=np.complex128)
    if len(coeffs) <= 1:
        return math.log(abs(float(coeffs[0].real))) if coeffs[0] != 0 else -math.inf
    roots = np.roots(coeffs)
    total = math.log(abs(leading_coeff(p))) if leading_coeff(p) else -math.inf
    for xi in roots:
        total += _quadratic_contribution_complex(complex(xi))
    return float(total)


def _roots_mpmath(poly: Poly, precision: int) -> list[mp.mpc]:
    p = to_poly(poly)
    coeffs = [mp.mpf(int(c)) for c in coeffs_high_to_low(p)]
    try:
        return [mp.mpc(r) for r in mp.polyroots(coeffs, maxsteps=300, error=False)]
    except Exception:
        roots = sp.Poly(p.as_expr(), p.gens[0]).nroots(n=precision, maxsteps=300)
        out: list[mp.mpc] = []
        for r in roots:
            out.append(mp.mpc(str(sp.re(r)), str(sp.im(r))))
        return out


def log_mahler_Q_z1mz(poly: Poly, precision: int = 80) -> mp.mpf:
    """High-precision numerical log Mahler measure of ``Q(z(1-z))``.

    If ``Q(X)=a prod(X-xi)``, then ``Q(z(1-z))`` has the roots of
    ``z^2-z+xi=0`` for each ``xi``.  The leading coefficient contribution
    ``log|a|`` is included; it is zero for the monic Doche polynomials.
    """

    old_dps = mp.mp.dps
    mp.mp.dps = precision
    try:
        p = to_poly(poly)
        if degree(p) <= 0:
            lc = abs(leading_coeff(p))
            return mp.log(lc) if lc else mp.ninf
        total = mp.log(abs(leading_coeff(p)))
        for xi in _roots_mpmath(p, precision):
            disc = mp.mpc(1) - 4 * xi
            root = mp.sqrt(disc)
            r_plus = (1 + root) / 2
            r_minus = (1 - root) / 2
            total += mp.log(max(mp.mpf(1), abs(r_plus)))
            total += mp.log(max(mp.mpf(1), abs(r_minus)))
        return +total
    finally:
        mp.mp.dps = old_dps


def log_mahler_product(
    factors: Iterable[tuple[Poly, int | float]], precision: int = 80, *, fast: bool = False
) -> float | mp.mpf:
    """Use additivity for products ``prod R_j^r_j``."""

    total: float | mp.mpf
    total = 0.0 if fast else mp.mpf("0")
    cache: dict[tuple[int, ...], float | mp.mpf] = {}
    from .polys import canonical_key

    for poly, multiplicity in factors:
        key = canonical_key(poly)
        if key not in cache:
            cache[key] = log_mahler_fast(poly) if fast else log_mahler_Q_z1mz(poly, precision)
        total += multiplicity * cache[key]
    return total

