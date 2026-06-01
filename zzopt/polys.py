"""Exact polynomial utilities for the Zhang-Zagier search.

The public API deliberately stays small and works with ``sympy.Poly`` in all
environments.  Sage is detected for users who want to extend the code in a Sage
session, but the fallback path is the supported test path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import prod
from typing import Iterable

import sympy as sp

try:  # pragma: no cover - Sage is optional in the CI/dev environment.
    from sageall import PolynomialRing, QQ as SAGE_QQ, ZZ as SAGE_ZZ  # type: ignore

    HAVE_SAGE = True
except Exception:  # pragma: no cover - normal path outside Sage.
    PolynomialRing = None
    SAGE_QQ = None
    SAGE_ZZ = None
    HAVE_SAGE = False


X = sp.Symbol("X")
Poly = sp.Poly


def poly_from_coeffs(coeffs: Iterable[int | sp.Integer]) -> Poly:
    """Create a polynomial from high-to-low integer coefficients."""

    coeff_list = [sp.Integer(c) for c in coeffs]
    if not coeff_list:
        coeff_list = [0]
    return sp.Poly.from_list(coeff_list, gens=X, domain=sp.ZZ)


def to_poly(poly: Poly | sp.Expr | Iterable[int]) -> Poly:
    if isinstance(poly, sp.Poly):
        if poly.gens == (X,) and poly.domain == sp.ZZ:
            return poly
        return sp.Poly(poly.as_expr(), X, domain=sp.ZZ)
    if isinstance(poly, (list, tuple)):
        return poly_from_coeffs(poly)
    return sp.Poly(poly, X, domain=sp.ZZ)


def degree(poly: Poly) -> int:
    return int(to_poly(poly).degree())


def leading_coeff(poly: Poly) -> int:
    return int(to_poly(poly).LC())


def coeffs_high_to_low(poly: Poly) -> list[int]:
    return [int(c) for c in to_poly(poly).all_coeffs()]


def coefficient_height(poly: Poly) -> int:
    coeffs = coeffs_high_to_low(poly)
    return max((abs(c) for c in coeffs), default=0)


def canonical_key(poly: Poly) -> tuple[int, ...]:
    return tuple(coeffs_high_to_low(primitive_part(poly, positive_lc=True)))


def multiply(*polys: Poly) -> Poly:
    if not polys:
        return poly_from_coeffs([1])
    expr = reduce(lambda a, b: a * to_poly(b).as_expr(), polys, sp.Integer(1))
    return sp.Poly(expr, X, domain=sp.ZZ)


def product_powers(powers: Iterable[tuple[Poly, int]]) -> Poly:
    out = poly_from_coeffs([1])
    for poly, exp in powers:
        if exp:
            out = multiply(out, power(poly, exp))
    return out


def add(a: Poly, b: Poly) -> Poly:
    return sp.Poly(to_poly(a).as_expr() + to_poly(b).as_expr(), X, domain=sp.ZZ)


def subtract(a: Poly, b: Poly) -> Poly:
    return sp.Poly(to_poly(a).as_expr() - to_poly(b).as_expr(), X, domain=sp.ZZ)


def power(poly: Poly, exponent: int) -> Poly:
    if exponent < 0:
        raise ValueError("negative polynomial powers are not supported")
    return sp.Poly(to_poly(poly).as_expr() ** int(exponent), X, domain=sp.ZZ)


def gcd(a: Poly, b: Poly) -> Poly:
    return sp.gcd(to_poly(a), to_poly(b))


def primitive_part(poly: Poly, *, positive_lc: bool = False) -> Poly:
    p = sp.Poly(to_poly(poly).as_expr(), X, domain=sp.ZZ)
    if p.is_zero:
        return p
    content, primitive = p.primitive()
    primitive = sp.Poly(primitive.as_expr(), X, domain=sp.ZZ)
    if positive_lc and primitive.LC() < 0:
        primitive = sp.Poly(-primitive.as_expr(), X, domain=sp.ZZ)
    return primitive


def factor_over_QQ(poly: Poly) -> list[tuple[Poly, int]]:
    """Factor over QQ and return primitive integer factors."""

    p = primitive_part(poly, positive_lc=True)
    if degree(p) <= 0:
        return []
    _, factors = sp.factor_list(p.as_expr(), X, domain=sp.QQ)
    out: list[tuple[Poly, int]] = []
    for fac_expr, mult in factors:
        fac = primitive_part(sp.Poly(fac_expr, X, domain=sp.QQ).clear_denoms()[1], positive_lc=True)
        out.append((fac, int(mult)))
    return out


def poly_to_string(poly: Poly) -> str:
    return str(to_poly(poly).as_expr())


def exact_degree_sum(factors: Iterable[tuple[Poly, int]]) -> int:
    return sum(degree(poly) * int(mult) for poly, mult in factors)


def named_product(names: Iterable[str], library: dict[str, Poly]) -> Poly:
    return multiply(*(library[name] for name in names))


@dataclass(frozen=True)
class FactorRecord:
    name: str
    coeffs: tuple[int, ...]
    degree: int

    @classmethod
    def from_poly(cls, name: str, poly: Poly) -> "FactorRecord":
        p = to_poly(poly)
        return cls(name=name, coeffs=tuple(coeffs_high_to_low(p)), degree=degree(p))

    def to_poly(self) -> Poly:
        return poly_from_coeffs(self.coeffs)


POLYS: dict[str, Poly] = {
    "P1": poly_from_coeffs([1, 0]),
    "P2": poly_from_coeffs([-1, 1]),
    "P3": poly_from_coeffs([1, 1, -2, 1]),
    "P4": poly_from_coeffs([1, -2, 4, -3, 1]),
    "P5": poly_from_coeffs([1, -2, 4, -7, 13, -16, 12, -5, 1]),
    "P6": poly_from_coeffs([1, -3, 8, -16, 26, -27, 17, -6, 1]),
    "P7": poly_from_coeffs(
        [1, -3, 8, -18, 36, -62, 97, -123, 114, -73, 31, -8, 1]
    ),
    "P8": poly_from_coeffs(
        [1, -3, 7, -14, 30, -58, 96, -123, 114, -73, 31, -8, 1]
    ),
    "P9": poly_from_coeffs(
        [
            1,
            -4,
            10,
            -17,
            26,
            -47,
            119,
            -298,
            592,
            -878,
            963,
            -780,
            464,
            -199,
            59,
            -11,
            1,
        ]
    ),
    "Q1": poly_from_coeffs(
        [
            1,
            -7,
            30,
            -97,
            269,
            -679,
            1612,
            -3618,
            7646,
            -15180,
            28457,
            -50741,
            86189,
            -138288,
            206152,
            -279897,
            339335,
            -360911,
            331775,
            -260367,
            172556,
            -95554,
            43677,
            -16221,
            4786,
            -1084,
            178,
            -19,
            1,
        ]
    ),
    "Q2": poly_from_coeffs(
        [
            1,
            -7,
            30,
            -96,
            255,
            -586,
            1212,
            -2360,
            4573,
            -9148,
            18749,
            -37783,
            71770,
            -124910,
            195848,
            -273368,
            335981,
            -359545,
            331349,
            -260271,
            172542,
            -95553,
            43677,
            -16221,
            4786,
            -1084,
            178,
            -19,
            1,
        ]
    ),
}


EXPECTED_DEGREES = {
    "P1": 1,
    "P2": 1,
    "P3": 3,
    "P4": 4,
    "P5": 8,
    "P6": 8,
    "P7": 12,
    "P8": 12,
    "P9": 16,
    "Q1": 28,
    "Q2": 28,
}

