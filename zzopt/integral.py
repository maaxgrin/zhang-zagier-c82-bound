"""Numerical integration of the Doche log-plus term."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import mpmath as mp
import numpy as np
from scipy import integrate

from .polys import Poly, coeffs_high_to_low, to_poly

TAU = 2.0 * math.pi
LOG_FLOOR = 1e-300


def chi_grid(n: int) -> np.ndarray:
    """Midpoint grid for chi(t), avoiding endpoints and exact singularities."""

    k = np.arange(n, dtype=np.float64)
    t = (k + 0.5) / float(n)
    u = np.exp(1j * TAU * t)
    return u * (1.0 - u)


def t_grid(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) / float(n)


def eval_poly_on_grid(poly: Poly, grid: np.ndarray) -> np.ndarray:
    coeffs = np.array(coeffs_high_to_low(to_poly(poly)), dtype=np.complex128)
    out = np.zeros_like(grid, dtype=np.complex128) + coeffs[0]
    for c in coeffs[1:]:
        out = out * grid + c
    return out


def log_abs_poly_on_chi(poly: Poly, grid: np.ndarray) -> np.ndarray:
    vals = eval_poly_on_grid(poly, grid)
    return np.log(np.maximum(np.abs(vals), LOG_FLOOR))


def log_abs_product_on_chi(factors: Iterable[tuple[Poly, int | float]], grid: np.ndarray) -> np.ndarray:
    out = np.zeros(len(grid), dtype=np.float64)
    for poly, mult in factors:
        out += float(mult) * log_abs_poly_on_chi(poly, grid)
    return out


def logplus_values(
    Q_factors: Sequence[tuple[Poly, int | float]],
    numerator_factors: Sequence[Poly],
    q: Sequence[float],
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_q = log_abs_product_on_chi(Q_factors, grid)
    a = np.zeros(len(grid), dtype=np.float64)
    for poly, qi in zip(numerator_factors, q):
        if qi:
            a += float(qi) * log_abs_poly_on_chi(poly, grid)
    g = a - b_q
    return np.maximum(0.0, g), g, b_q


def integral_logplus(
    Q_factors: Sequence[tuple[Poly, int | float]],
    numerator_factors: Sequence[Poly],
    q: Sequence[float],
    grid_size: int = 8192,
    *,
    grid: np.ndarray | None = None,
) -> float:
    if grid is None:
        grid = chi_grid(grid_size)
    values, _, _ = logplus_values(Q_factors, numerator_factors, q, grid)
    return float(np.mean(values))


def _mp_eval_poly(poly: Poly, x: mp.mpc) -> mp.mpc:
    coeffs = coeffs_high_to_low(to_poly(poly))
    out = mp.mpc(coeffs[0])
    for c in coeffs[1:]:
        out = out * x + c
    return out


def _mp_log_abs_poly(poly: Poly, x: mp.mpc) -> mp.mpf:
    val = abs(_mp_eval_poly(poly, x))
    if val == 0:
        return mp.ninf
    return mp.log(val)


def _mp_chi(t: mp.mpf) -> mp.mpc:
    u = mp.e ** (2 * mp.pi * 1j * t)
    return u * (1 - u)


def integral_logplus_mpmath_panel(
    Q_factors: Sequence[tuple[Poly, int | float]],
    numerator_factors: Sequence[Poly],
    q: Sequence[float],
    *,
    precision: int = 80,
    panels: int = 256,
) -> mp.mpf:
    """Composite high-precision panel quadrature.

    This is a refinement tool, not an interval certificate.  Midpoint panels
    keep endpoint logarithmic singularities out of direct evaluations.
    """

    old_dps = mp.mp.dps
    mp.mp.dps = precision
    try:
        total = mp.mpf("0")
        for j in range(panels):
            a = mp.mpf(j) / panels
            b = mp.mpf(j + 1) / panels
            eps = (b - a) * mp.mpf("1e-20")

            def f(t: mp.mpf) -> mp.mpf:
                x = _mp_chi(t)
                g = mp.mpf("0")
                for poly, qi in zip(numerator_factors, q):
                    if qi:
                        g += mp.mpf(str(qi)) * _mp_log_abs_poly(poly, x)
                for poly, mult in Q_factors:
                    g -= mp.mpf(str(mult)) * _mp_log_abs_poly(poly, x)
                if g <= 0 or g == mp.ninf:
                    return mp.mpf("0")
                return g

            total += mp.quad(f, [a + eps, (a + b) / 2, b - eps])
        return +total
    finally:
        mp.mp.dps = old_dps


def integral_logplus_adaptive(
    Q_factors: Sequence[tuple[Poly, int | float]],
    numerator_factors: Sequence[Poly],
    q: Sequence[float],
    *,
    epsabs: float = 1e-10,
    limit: int = 500,
    pilot_grid: int = 8192,
    expand_cells: int = 4,
) -> tuple[float, float]:
    """Independent adaptive SciPy quadrature for comparison in refinement.

    A blind call to ``quad`` can miss very narrow positive regions of the
    log-plus integrand.  We first locate active cells on a midpoint grid, expand
    them slightly, and integrate those panels separately.  This is still not an
    interval certificate.
    """

    def f(t: float) -> float:
        u = complex(math.cos(TAU * t), math.sin(TAU * t))
        x = u * (1.0 - u)
        g = 0.0
        for poly, qi in zip(numerator_factors, q):
            if qi:
                coeffs = coeffs_high_to_low(poly)
                val = complex(coeffs[0])
                for c in coeffs[1:]:
                    val = val * x + c
                g += float(qi) * math.log(max(abs(val), LOG_FLOOR))
        for poly, mult in Q_factors:
            coeffs = coeffs_high_to_low(poly)
            val = complex(coeffs[0])
            for c in coeffs[1:]:
                val = val * x + c
            g -= float(mult) * math.log(max(abs(val), LOG_FLOOR))
        return max(0.0, g)

    if pilot_grid <= 0:
        value, error = integrate.quad(f, 0.0, 1.0, epsabs=epsabs, epsrel=epsabs, limit=limit, points=[0.5])
        return float(value), float(error)

    pilot = chi_grid(pilot_grid)
    _, g_values, _ = logplus_values(Q_factors, numerator_factors, q, pilot)
    active = g_values > 0
    if not active.any():
        value, error = integrate.quad(f, 0.0, 1.0, epsabs=epsabs, epsrel=epsabs, limit=limit, points=[0.5])
        return float(value), float(error)

    intervals: list[tuple[float, float]] = []
    indices = np.flatnonzero(active)
    start = int(indices[0])
    prev = int(indices[0])
    for idx in map(int, indices[1:]):
        if idx == prev + 1:
            prev = idx
            continue
        a_idx = max(0, start - expand_cells)
        b_idx = min(pilot_grid, prev + 1 + expand_cells)
        intervals.append((a_idx / pilot_grid, b_idx / pilot_grid))
        start = prev = idx
    a_idx = max(0, start - expand_cells)
    b_idx = min(pilot_grid, prev + 1 + expand_cells)
    intervals.append((a_idx / pilot_grid, b_idx / pilot_grid))

    # Merge overlaps after expansion.
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    total = 0.0
    total_error = 0.0
    for a, b in merged:
        if b <= a:
            continue
        mid = (a + b) / 2.0
        value, error = integrate.quad(
            f,
            a,
            b,
            epsabs=epsabs,
            epsrel=epsabs,
            limit=limit,
            points=[mid],
        )
        total += float(value)
        total_error += float(error)
    return total, total_error
