"""Exhaustive numerical enumeration of a finite denominator family."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Sequence

from .candidate_generation import build_initial_library
from .mahler import log_mahler_fast
from .polys import Poly, degree
from .search import State, evaluate_state, validate_doche_baseline


def _parse_names(text: str) -> list[str]:
    if not text.strip():
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_mults(text: str, names: Sequence[str], default: int) -> dict[str, int]:
    out = {name: default for name in names}
    if not text.strip():
        return out
    for part in text.split(","):
        if not part.strip():
            continue
        name, value = part.split(":", 1)
        out[name.strip()] = int(value)
    return out


def enumerate_states(
    *,
    required: Sequence[str],
    optional: Sequence[str],
    max_mult: dict[str, int],
    library: dict[str, Poly],
    max_degree: int,
) -> list[State]:
    base_items = [(name, 1) for name in required]
    base = State.from_items(base_items, parent="finite_required")
    if base.degree(library) > max_degree:
        return []

    ranges = [range(max_mult.get(name, 1) + 1) for name in optional]
    states: list[State] = []
    for exps in itertools.product(*ranges):
        items = list(base_items) + [(name, exp) for name, exp in zip(optional, exps) if exp]
        state = State.from_items(items, parent="finite_exhaustive")
        if 0 < state.degree(library) <= max_degree:
            states.append(state)
    # Canonicalize in case required/optional overlap.
    unique = {state.key(): state for state in states}
    return list(unique.values())


def run_exhaustive(args: argparse.Namespace) -> list[dict]:
    library = build_initial_library()
    required = _parse_names(args.required)
    optional = _parse_names(args.optional)
    numerator_names = _parse_names(args.numerator)

    unknown = sorted((set(required) | set(optional) | set(numerator_names)) - set(library))
    if unknown:
        raise KeyError(f"Unknown factors: {', '.join(unknown)}")

    validate_doche_baseline(library, args.baseline_grid or min(args.grid, 16384))

    max_mult = _parse_mults(args.multiplicities, optional, args.default_max_mult)
    states = enumerate_states(
        required=required,
        optional=optional,
        max_mult=max_mult,
        library=library,
        max_degree=args.max_degree,
    )
    logM_by_name = {name: log_mahler_fast(poly) for name, poly in library.items()}

    manifest = {
        "schema": 1,
        "record_type": "finite_space_manifest",
        "status": "numeric_only",
        "description": (
            "Exhaustive enumeration of this finite denominator family. "
            "LP optimality is only for the discretized grid; this is not a proof."
        ),
        "required": required,
        "optional": optional,
        "max_multiplicities": max_mult,
        "numerator": numerator_names,
        "max_degree": args.max_degree,
        "grid_size": args.grid,
        "max_den": args.max_den,
        "state_count": len(states),
        "certified": False,
    }

    records: list[dict] = []
    for index, state in enumerate(states, 1):
        rec = evaluate_state(
            state,
            library,
            numerator_names=numerator_names,
            grid_size=args.grid,
            max_den=args.max_den,
            label="finite_exhaustive",
            logM_by_name=logM_by_name,
        )
        rec["finite_space"] = {
            "required": required,
            "optional": optional,
            "max_multiplicities": max_mult,
            "state_count": len(states),
            "index": index,
        }
        records.append(rec)
        if args.progress and (index % args.progress == 0 or index == len(states)):
            best = min(records, key=lambda r: r["bound_log"])
            print(
                f"{index}/{len(states)} best={best['bound_log']:.10f} "
                f"degQ={best['degQ']}",
                flush=True,
            )

    records.sort(key=lambda r: r["bound_log"])
    for rank, rec in enumerate(records, 1):
        rec["finite_rank"] = rank
        rec["best_in_finite_space"] = rank == 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, sort_keys=True) + "\n")
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required", default="Q1,Q2")
    parser.add_argument("--optional", default="R0,R2,P1,P2,P3,P4,P5,P6,P7,P8,P9")
    parser.add_argument("--numerator", default="P1,P2,P3,P4,P5,P6,P7,P8,P9")
    parser.add_argument("--max-degree", type=int, default=220)
    parser.add_argument("--grid", type=int, default=4096)
    parser.add_argument("--baseline-grid", type=int, default=16384)
    parser.add_argument("--max-den", type=int, default=1000000)
    parser.add_argument("--default-max-mult", type=int, default=1)
    parser.add_argument(
        "--multiplicities",
        default="",
        help="Comma-separated overrides such as P7:2,P9:2.",
    )
    parser.add_argument("--progress", type=int, default=128)
    parser.add_argument("--out", default="runs/exhaustive.jsonl")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = run_exhaustive(args)
    for idx, rec in enumerate(records[:20], 1):
        den = " * ".join(f"{f['name']}^{f['multiplicity']}" for f in rec["denominator"])
        print(
            f"{idx:02d} bound_log={rec['bound_log']:.10f} "
            f"exp={math.exp(rec['bound_log']):.9f} degQ={rec['degQ']} Q={den}"
        )


if __name__ == "__main__":
    main()
