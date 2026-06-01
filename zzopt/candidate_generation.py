"""Generation and scoring of denominator factors."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np

from .integral import log_abs_poly_on_chi
from .mahler import log_mahler_fast
from .polys import (
    POLYS,
    Poly,
    add,
    canonical_key,
    coefficient_height,
    degree,
    factor_over_QQ,
    gcd,
    multiply,
    power,
    primitive_part,
    subtract,
)


@dataclass(frozen=True)
class GeneratedFactor:
    name: str
    poly: Poly
    source: str
    raw_score: float


def build_initial_library() -> dict[str, Poly]:
    lib = dict(POLYS)
    bridge = multiply(power(lib["P1"], 5), power(lib["P2"], 5), lib["P4"])
    lib["R0"] = primitive_part(subtract(lib["Q1"], multiply(bridge, lib["P8"])), positive_lc=True)
    lib["R2"] = primitive_part(add(lib["Q2"], multiply(bridge, lib["P7"])), positive_lc=True)
    return lib


def denominator_seed_factors(seed: str, library: dict[str, Poly]) -> list[tuple[str, int]]:
    if seed == "doche":
        return [("Q1", 1), ("Q2", 1)]
    if seed == "star":
        return [("Q1", 1), ("Q2", 1), ("R0", 1), ("R2", 1), ("P7", 1), ("P9", 1)]
    if seed == "q12p7":
        return [("Q1", 1), ("Q2", 1), ("P7", 1)]
    if seed == "q12p9":
        return [("Q1", 1), ("Q2", 1), ("P9", 1)]
    if seed == "q12p7p9":
        return [("Q1", 1), ("Q2", 1), ("P7", 1), ("P9", 1)]
    raise KeyError(seed)


def enumerate_products_near_degree(
    basis: Sequence[tuple[str, Poly]],
    target_degree: int,
    *,
    window: int = 4,
    max_exp: int = 8,
    max_count: int = 2000,
) -> list[tuple[str, Poly]]:
    """Enumerate products of basis factors whose degree is close to target."""

    out: list[tuple[str, Poly]] = []
    exps = [range(max_exp + 1) for _ in basis]
    for exp_tuple in product(*exps):
        deg = sum(e * degree(poly) for e, (_, poly) in zip(exp_tuple, basis))
        if abs(deg - target_degree) > window:
            continue
        if deg == 0:
            continue
        factors = [power(poly, e) for e, (_, poly) in zip(exp_tuple, basis) if e]
        label_bits = [f"{name}^{e}" for e, (name, _) in zip(exp_tuple, basis) if e]
        out.append(("*".join(label_bits), multiply(*factors)))
        if len(out) >= max_count:
            break
    return out


def generate_doche_type_factors(
    library: dict[str, Poly],
    *,
    base_names: Sequence[str] = ("Q1", "Q2", "R0", "R2"),
    product_basis_names: Sequence[str] = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"),
    degree_window: int = 3,
    max_exp: int = 6,
    max_factor_degree: int = 80,
    max_coeff_height: int = 1_000_000_000,
    max_products_per_base: int = 300,
    known_keys: set[tuple[int, ...]] | None = None,
) -> list[GeneratedFactor]:
    known = set(known_keys or set())
    for poly in library.values():
        known.add(canonical_key(poly))
    product_basis = [(name, library[name]) for name in product_basis_names if name in library]
    generated: list[GeneratedFactor] = []
    counter = 1
    for base_name in base_names:
        if base_name not in library:
            continue
        F = library[base_name]
        products = enumerate_products_near_degree(
            product_basis,
            degree(F),
            window=degree_window,
            max_exp=max_exp,
            max_count=max_products_per_base,
        )
        for g_label, G in products:
            for sign, H in (("plus", add(F, G)), ("minus", subtract(F, G))):
                H = primitive_part(H, positive_lc=True)
                if degree(H) <= 0:
                    continue
                try:
                    factors = factor_over_QQ(H)
                except Exception:
                    continue
                for fac, _mult in factors:
                    deg = degree(fac)
                    if deg < 2 or deg > max_factor_degree:
                        continue
                    if coefficient_height(fac) > max_coeff_height:
                        continue
                    key = canonical_key(fac)
                    if key in known:
                        continue
                    known.add(key)
                    name = f"G{counter:04d}"
                    counter += 1
                    generated.append(
                        GeneratedFactor(
                            name=name,
                            poly=fac,
                            source=f"{base_name} {sign} ({g_label})",
                            raw_score=log_mahler_fast(fac) / deg,
                        )
                    )
    generated.sort(key=lambda item: item.raw_score)
    return generated


def marginal_score(
    poly: Poly,
    *,
    grid: np.ndarray,
    active_mask: np.ndarray,
    lambda_value: float,
) -> float:
    if active_mask.any():
        integral_e = float(np.mean(log_abs_poly_on_chi(poly, grid)[active_mask]))
    else:
        integral_e = 0.0
    return log_mahler_fast(poly) - integral_e - lambda_value * degree(poly)


def gcd_ok_with_denominator(poly: Poly, denominator: Sequence[tuple[Poly, int]]) -> bool:
    return all(degree(gcd(poly, den_poly)) == 0 for den_poly, _ in denominator)

