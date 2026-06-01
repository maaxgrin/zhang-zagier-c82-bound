"""Beam/greedy numerical search for Doche-type bounds."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .candidate_generation import (
    build_initial_library,
    denominator_seed_factors,
    generate_doche_type_factors,
    marginal_score,
)
from .integral import chi_grid, log_abs_product_on_chi, logplus_values
from .lp_opt import optimize_exponents_lp
from .mahler import log_mahler_product
from .polys import (
    POLYS,
    FactorRecord,
    Poly,
    canonical_key,
    coeffs_high_to_low,
    degree,
    gcd,
    multiply,
)


BASELINE_Q = [("Q1", 1), ("Q2", 1)]
BASELINE_NUMERATOR = ["P1", "P2", "P4", "P6", "P8"]
BASELINE_Q_VALUES = [13.1, 10.6, 3.2, 1.15, 0.24]
STAR_Q_VALUES = {
    "P1": 26.5179763725,
    "P2": 23.8085238104,
    "P3": 0.9725872822,
    "P4": 4.5133890152,
    "P5": 0.0536766104,
    "P6": 4.1833486702,
    "P8": 1.6692966840,
}


@dataclass(frozen=True)
class State:
    factors: tuple[tuple[str, int], ...]
    parent: str = ""

    @classmethod
    def from_items(cls, items: Iterable[tuple[str, int]], parent: str = "") -> "State":
        c: Counter[str] = Counter()
        for name, mult in items:
            if mult:
                c[name] += int(mult)
        return cls(tuple(sorted((name, mult) for name, mult in c.items() if mult > 0)), parent)

    def degree(self, library: dict[str, Poly]) -> int:
        return sum(degree(library[name]) * mult for name, mult in self.factors)

    def q_factors(self, library: dict[str, Poly]) -> list[tuple[Poly, int]]:
        return [(library[name], mult) for name, mult in self.factors]

    def key(self) -> tuple[tuple[str, int], ...]:
        return self.factors


def _serial_factor(name: str, poly: Poly, multiplicity: int | None = None) -> dict:
    rec = FactorRecord.from_poly(name, poly)
    data = {"name": rec.name, "degree": rec.degree, "coeffs": list(rec.coeffs)}
    if multiplicity is not None:
        data["multiplicity"] = multiplicity
    return data


def admissible_numerators(
    numerator_names: Sequence[str],
    denominator: Sequence[tuple[Poly, int]],
    library: dict[str, Poly],
) -> list[str]:
    out = []
    for name in numerator_names:
        poly = library[name]
        if all(degree(gcd(poly, dpoly)) == 0 for dpoly, _ in denominator):
            out.append(name)
    return out


def evaluate_state(
    state: State,
    library: dict[str, Poly],
    *,
    numerator_names: Sequence[str],
    grid_size: int,
    max_den: int,
    status: str = "numeric_only",
    label: str = "",
    logM_by_name: dict[str, float] | None = None,
) -> dict:
    q_factors = state.q_factors(library)
    degQ = state.degree(library)
    if logM_by_name is None:
        logM = float(log_mahler_product(q_factors, fast=True))
    else:
        logM = sum(int(mult) * logM_by_name[name] for name, mult in state.factors)
    usable_names = admissible_numerators(numerator_names, q_factors, library)
    numerator = [library[name] for name in usable_names]
    lp = optimize_exponents_lp(
        q_factors,
        numerator,
        grid_size=grid_size,
        max_den=max_den,
        logM=logM,
    )
    return {
        "schema": 1,
        "status": status,
        "label": label or "lp_state",
        "success": lp.success,
        "message": lp.message,
        "bound_log": lp.bound_log,
        "bound_exp": math.exp(lp.bound_log) if math.isfinite(lp.bound_log) else float("inf"),
        "degQ": lp.degQ,
        "degA": lp.degA,
        "degA_rational": lp.degA_rational,
        "logM": lp.logM,
        "integral": lp.integral,
        "grid_size": grid_size,
        "denominator": [_serial_factor(name, library[name], mult) for name, mult in state.factors],
        "numerator": [
            {
                **_serial_factor(name, library[name]),
                "q": q,
                "q_rational": qr,
            }
            for name, q, qr in zip(usable_names, lp.q, lp.q_rational)
            if q > 1e-12
        ],
        "all_numerator_names": usable_names,
        "parent": state.parent,
    }


def evaluate_fixed_q(
    denominator_items: Sequence[tuple[str, int]],
    q_by_name: dict[str, float],
    library: dict[str, Poly],
    *,
    grid_size: int,
    label: str,
) -> dict:
    from .integral import integral_logplus

    state = State.from_items(denominator_items)
    q_factors = state.q_factors(library)
    numerator_names = [name for name, q in q_by_name.items() if q > 0]
    numerator = [library[name] for name in numerator_names]
    q = [q_by_name[name] for name in numerator_names]
    degQ = state.degree(library)
    degA = sum(degree(library[name]) * q_by_name[name] for name in numerator_names)
    logM = float(log_mahler_product(q_factors, fast=True))
    integral = integral_logplus(q_factors, numerator, q, grid_size=grid_size)
    bound = (logM + integral) / degQ
    return {
        "schema": 1,
        "status": "numeric_only",
        "label": label,
        "success": True,
        "bound_log": bound,
        "bound_exp": math.exp(bound),
        "degQ": degQ,
        "degA": degA,
        "degA_rational": "",
        "logM": logM,
        "integral": integral,
        "grid_size": grid_size,
        "denominator": [_serial_factor(name, library[name], mult) for name, mult in state.factors],
        "numerator": [
            {
                **_serial_factor(name, library[name]),
                "q": q_by_name[name],
                "q_rational": "",
            }
            for name in numerator_names
        ],
        "parent": "",
    }


def state_from_record(record: dict, *, parent: str = "") -> State:
    return State.from_items(
        [(item["name"], int(item.get("multiplicity", 1))) for item in record["denominator"]],
        parent=parent or record.get("label", ""),
    )


def validate_doche_baseline(library: dict[str, Poly], grid_size: int) -> dict:
    baseline = evaluate_fixed_q(
        BASELINE_Q,
        dict(zip(BASELINE_NUMERATOR, BASELINE_Q_VALUES)),
        library,
        grid_size=grid_size,
        label="doche_baseline_fixed_q",
    )
    if not (0.2535 <= baseline["bound_log"] <= 0.2555):
        raise RuntimeError(
            f"Doche baseline normalization failed: {baseline['bound_log']:.10f} "
            "not in [0.2535, 0.2555]"
        )
    return baseline


def _write_jsonl(path: Path, records: Iterable[dict], mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def search(args: argparse.Namespace) -> list[dict]:
    library = build_initial_library()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.append:
        out.unlink()

    baseline = validate_doche_baseline(library, args.baseline_grid or args.grid)
    _write_jsonl(out, [baseline])

    star = evaluate_fixed_q(
        denominator_seed_factors("star", library),
        STAR_Q_VALUES,
        library,
        grid_size=args.grid,
        label="provided_seed_star_fixed_q",
    )
    _write_jsonl(out, [star])

    generated = generate_doche_type_factors(
        library,
        degree_window=args.generation_degree_window,
        max_exp=args.generation_max_exp,
        max_factor_degree=args.max_factor_degree,
        max_coeff_height=args.max_coeff_height,
        max_products_per_base=args.max_products_per_base,
    )
    for item in generated[: args.generated_keep]:
        library[item.name] = item.poly

    numerator_names = [f"P{i}" for i in range(1, 10) if f"P{i}" in library]
    initial_additions = [f"P{i}" for i in range(1, 10)] + ["Q1", "Q2", "R0", "R2"]
    candidate_additions = initial_additions + [g.name for g in generated[: args.generated_keep]]
    candidate_additions = [name for name in candidate_additions if name in library]

    seeds = ["doche", "star", "q12p7", "q12p9", "q12p7p9"]
    beam = [
        State.from_items(denominator_seed_factors(seed, library), parent=f"seed:{seed}")
        for seed in seeds
        if State.from_items(denominator_seed_factors(seed, library)).degree(library) <= args.max_degree
    ]

    seen: set[tuple[tuple[str, int], ...]] = set()
    all_records: list[dict] = [baseline, star]
    for round_idx in range(args.rounds):
        round_records = []
        for state in beam:
            if state.key() in seen:
                continue
            seen.add(state.key())
            rec = evaluate_state(
                state,
                library,
                numerator_names=numerator_names,
                grid_size=args.grid,
                max_den=args.max_den,
                label=f"round_{round_idx}",
            )
            round_records.append(rec)
        round_records.sort(key=lambda r: r["bound_log"])
        _write_jsonl(out, round_records)
        all_records.extend(round_records)

        next_states: dict[tuple[tuple[str, int], ...], State] = {}
        top_records = round_records[: args.beam_width]

        grid = chi_grid(max(512, min(args.grid, 4096)))
        for rec in top_records:
            state = state_from_record(rec, parent=rec["label"])
            q_factors = state.q_factors(library)
            used_num = [library[n["name"]] for n in rec.get("numerator", [])]
            q = [float(n["q"]) for n in rec.get("numerator", [])]
            if used_num:
                _, g_values, _ = logplus_values(q_factors, used_num, q, grid)
                active = g_values > 0
            else:
                bq = log_abs_product_on_chi(q_factors, grid)
                active = (-bq) > 0
            scored = []
            for name in candidate_additions:
                new_degree = state.degree(library) + degree(library[name])
                if new_degree > args.max_degree:
                    continue
                score = marginal_score(
                    library[name],
                    grid=grid,
                    active_mask=active,
                    lambda_value=float(rec["bound_log"]),
                )
                scored.append((score, name))
            scored.sort(key=lambda x: x[0])
            for _score, name in scored[: args.children_per_state]:
                child = State.from_items(list(state.factors) + [(name, 1)], parent=f"add:{name}")
                if child.key() not in seen:
                    next_states[child.key()] = child
            for name, mult in state.factors:
                if mult <= 0:
                    continue
                reduced = list(state.factors)
                reduced.remove((name, mult))
                if mult > 1:
                    reduced.append((name, mult - 1))
                child = State.from_items(reduced, parent=f"remove:{name}")
                if child.factors and child.key() not in seen:
                    next_states[child.key()] = child

        beam = list(next_states.values())[: args.beam_width]
        if not beam:
            break

    return sorted(all_records, key=lambda r: r["bound_log"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, default=240)
    parser.add_argument("--grid", type=int, default=8192)
    parser.add_argument("--baseline-grid", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--children-per-state", type=int, default=5)
    parser.add_argument("--max-factor-degree", type=int, default=80)
    parser.add_argument("--max-coeff-height", type=int, default=1_000_000_000)
    parser.add_argument("--max-den", type=int, default=100000)
    parser.add_argument("--generation-degree-window", type=int, default=3)
    parser.add_argument("--generation-max-exp", type=int, default=6)
    parser.add_argument("--max-products-per-base", type=int, default=300)
    parser.add_argument("--generated-keep", type=int, default=30)
    parser.add_argument("--out", default="runs/search.jsonl")
    parser.add_argument("--append", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    best = search(args)[:20]
    for idx, rec in enumerate(best, 1):
        den = " * ".join(f"{f['name']}^{f['multiplicity']}" for f in rec["denominator"])
        print(
            f"{idx:02d} bound_log={rec['bound_log']:.10f} exp={rec['bound_exp']:.9f} "
            f"degQ={rec['degQ']} status={rec['status']} Q={den}"
        )


if __name__ == "__main__":
    main()
