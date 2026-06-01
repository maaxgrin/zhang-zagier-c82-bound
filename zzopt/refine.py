"""Refine promising JSONL candidates numerically."""

from __future__ import annotations

import argparse
import copy
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from .integral import integral_logplus, integral_logplus_adaptive, integral_logplus_mpmath_panel
from .mahler import log_mahler_Q_z1mz
from .polys import Poly, canonical_key, degree, gcd, poly_from_coeffs


def _poly_from_record(record: dict) -> Poly:
    return poly_from_coeffs(record["coeffs"])


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _q_fracs(numerator: list[dict], max_den: int) -> list[Fraction]:
    return [Fraction(max(0.0, float(item.get("q", 0.0)))).limit_denominator(max_den) for item in numerator]


def _log_mahler_product_cached(
    denominator: list[tuple[Poly, int]],
    *,
    precision: int,
    cache: dict[tuple[tuple[int, ...], int], float] | None,
) -> float:
    total = 0.0
    for poly, mult in denominator:
        key = (canonical_key(poly), precision)
        if cache is not None and key in cache:
            value = cache[key]
        else:
            value = float(log_mahler_Q_z1mz(poly, precision=precision))
            if cache is not None:
                cache[key] = value
        total += int(mult) * value
    return total


def refine_record(
    record: dict,
    *,
    grids: Sequence[int],
    max_den: int,
    precision: int,
    adaptive: bool,
    mp_panels: int = 0,
    mahler_cache: dict[tuple[tuple[int, ...], int], float] | None = None,
) -> dict:
    denominator = [(_poly_from_record(item), int(item.get("multiplicity", 1))) for item in record["denominator"]]
    numerator_records = [item for item in record.get("numerator", []) if float(item.get("q", 0.0)) > 0]
    numerator = [_poly_from_record(item) for item in numerator_records]
    q_frac = _q_fracs(numerator_records, max_den)
    degQ = sum(degree(poly) * mult for poly, mult in denominator)
    degA = sum(Fraction(degree(poly), 1) * qi for poly, qi in zip(numerator, q_frac))
    if degA >= degQ and degA > 0:
        scale = Fraction(degQ * max_den - 1, max_den) / degA
        q_frac = [qi * scale for qi in q_frac]
        degA = sum(Fraction(degree(poly), 1) * qi for poly, qi in zip(numerator, q_frac))
    q = [float(x) for x in q_frac]
    logM = _log_mahler_product_cached(denominator, precision=precision, cache=mahler_cache)
    grid_integrals = {
        str(grid): integral_logplus(denominator, numerator, q, grid_size=grid)
        for grid in grids
    }
    integral = grid_integrals[str(max(grids))]
    adaptive_result = None
    if adaptive:
        try:
            value, error = integral_logplus_adaptive(
                denominator,
                numerator,
                q,
                epsabs=1e-9,
                pilot_grid=max(grids),
            )
            adaptive_result = {"value": value, "error_estimate": error}
            integral = max(integral, value)
        except Exception as exc:
            adaptive_result = {"error": repr(exc)}
    mp_panel_result = None
    if mp_panels > 0:
        try:
            panels = mp_panels
            mp_value = integral_logplus_mpmath_panel(
                denominator,
                numerator,
                q,
                precision=precision,
                panels=panels,
            )
            mp_panel_result = {"value": float(mp_value), "panels": panels}
            integral = max(integral, float(mp_value))
        except Exception as exc:
            mp_panel_result = {"error": repr(exc), "panels": mp_panels}
    else:
        mp_panel_result = {"skipped": True, "reason": "mp_panels=0"}
    gcd_ok = all(degree(gcd(spoly, qpoly)) == 0 for spoly in numerator for qpoly, _ in denominator)
    refined = copy.deepcopy(record)
    refined.update(
        {
            "status": "numeric_refined",
            "refined_from_bound_log": record.get("bound_log"),
            "bound_log": (logM + integral) / degQ,
            "bound_exp": math.exp((logM + integral) / degQ),
            "degQ": degQ,
            "degA": float(degA),
            "degA_rational": str(degA),
            "logM": logM,
            "integral": integral,
            "integral_grid": grid_integrals,
            "integral_adaptive": adaptive_result,
            "integral_mpmath_panel": mp_panel_result,
            "q_max_den": max_den,
            "gcd_ok": gcd_ok,
            "degree_ok": degA < degQ,
            "certified": False,
            "certification_note": "No interval certificate was computed; this is numeric_refined only.",
        }
    )
    for item, qi in zip(refined.get("numerator", []), q_frac):
        if float(item.get("q", 0.0)) > 0:
            item["q"] = float(qi)
            item["q_rational"] = str(qi)
    return refined


def write_jsonl(path: str | Path, records: Sequence[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="runs/search.jsonl")
    parser.add_argument("--out", default="runs/refined.jsonl")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--grids", default="16384,32768,65536")
    parser.add_argument("--max-den", type=int, default=1_000_000)
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument(
        "--mp-panels",
        type=int,
        default=0,
        help="Optional mpmath panel quadrature count; 0 skips this slow cross-check.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = [r for r in load_jsonl(args.input) if r.get("success", True) and math.isfinite(r.get("bound_log", math.inf))]
    records.sort(key=lambda r: r["bound_log"])
    grids = [int(x) for x in args.grids.split(",") if x.strip()]
    mahler_cache: dict[tuple[tuple[int, ...], int], float] = {}
    refined = [
        refine_record(
            record,
            grids=grids,
            max_den=args.max_den,
            precision=args.precision,
            adaptive=args.adaptive,
            mp_panels=args.mp_panels,
            mahler_cache=mahler_cache,
        )
        for record in records[: args.top]
    ]
    refined.sort(key=lambda r: r["bound_log"])
    write_jsonl(args.out, refined)
    for idx, rec in enumerate(refined, 1):
        print(
            f"{idx:02d} bound_log={rec['bound_log']:.10f} exp={rec['bound_exp']:.9f} "
            f"degQ={rec['degQ']} status={rec['status']} gcd_ok={rec['gcd_ok']}"
        )


if __name__ == "__main__":
    main()
