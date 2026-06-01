"""Certification helpers for numerical candidates.

The Sage export is still provided, but this module now also contains a small
Arb-backed interval upper-bound checker using ``python-flint`` when available.
It is deliberately conservative: ``certified=True`` is returned only when all
panels are bounded and the final Arb upper bound is below the requested target.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from .polys import Poly, canonical_key, coeffs_high_to_low, degree, leading_coeff, poly_from_coeffs

try:  # pragma: no cover - optional dependency availability is environment-specific.
    from flint import acb, acb_poly, arb, ctx

    HAVE_FLINT = True
except Exception:  # pragma: no cover
    acb = None
    acb_poly = None
    arb = None
    ctx = None
    HAVE_FLINT = False


def export_candidate_for_sage_arb(candidate: dict, path: str | Path) -> Path:
    """Export a candidate and a Sage skeleton for interval certification."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(candidate, indent=2, sort_keys=True)
    script = f'''# Auto-generated certification skeleton.
# This script is not a proof yet.  It must be completed with Arb-backed
# interval arithmetic before any candidate can be marked certified.

import json

candidate = json.loads(r"""{payload}""")

# TODO:
# 1. Build exact QQ polynomials from candidate["denominator"] and ["numerator"].
# 2. Isolate roots of Q(chi(t)) on t in [0, 1].
# 3. Split [0, 1] at certified isolating intervals and sign-change intervals of
#    G(t) = sum q_i log|S_i(chi(t))| - log|Q(chi(t))|.
# 4. Treat logarithmic singularities with explicit local majorants.
# 5. Integrate max(0, G(t)) using Arb intervals and combine with an interval
#    upper bound for log M(Q(z(1-z))).
# 6. Only then may the caller set certified=True.

print("certified=False")
'''
    out.write_text(script, encoding="utf-8")
    return out


def isolate_roots_placeholder(candidate: dict) -> dict:
    return {
        "certified": False,
        "root_isolation_done": False,
        "note": "Root isolation on Q(chi(t)) is not implemented in this stub.",
    }


def _arb_fraction(value: Fraction | int | str) -> "arb":
    if isinstance(value, Fraction):
        return arb(value.numerator) / value.denominator
    if isinstance(value, int):
        return arb(value)
    return arb(str(value))


def _poly_from_record(record: dict) -> Poly:
    return poly_from_coeffs(record["coeffs"])


def _q_fraction(record: dict) -> Fraction:
    q_rat = record.get("q_rational")
    if q_rat:
        return Fraction(q_rat)
    return Fraction(str(record.get("q", 0))).limit_denominator(10_000_000)


def _acb_eval_poly(poly: Poly, x: "acb") -> "acb":
    coeffs = coeffs_high_to_low(poly)
    out = acb(coeffs[0])
    for coeff in coeffs[1:]:
        out = out * x + coeff
    return out


def _chi_interval(t: "arb") -> "acb":
    theta = 2 * arb.pi() * t
    u = acb(theta.cos(), theta.sin())
    return u * (1 - u)


def _poly_roots(poly: Poly) -> list["acb"]:
    coeffs = list(reversed(coeffs_high_to_low(poly)))
    p = acb_poly([acb(c) for c in coeffs])
    tol = 2.0 ** (-(ctx.prec // 2))
    roots = p.complex_roots(tol=tol, maxprec=max(256, 4 * ctx.prec))
    if len(roots) != degree(poly):
        raise RuntimeError(f"expected {degree(poly)} roots, got {len(roots)}")
    return roots


def _root_data(poly: Poly, cache: dict[tuple[int, ...], dict] | None = None) -> dict:
    key = canonical_key(poly)
    if cache is not None and key in cache:
        return cache[key]
    data = {
        "degree": degree(poly),
        "leading_coeff_abs": abs(leading_coeff(poly)),
        "roots": _poly_roots(poly),
    }
    if cache is not None:
        cache[key] = data
    return data


def _log_abs_upper(poly: Poly, x: "acb") -> "arb":
    abs_interval = abs(_acb_eval_poly(poly, x))
    upper = abs_interval.upper()
    return upper.log().upper()


def _log_abs_lower(poly: Poly, x: "acb") -> "arb | None":
    abs_interval = abs(_acb_eval_poly(poly, x))
    lower = abs_interval.lower()
    if not (lower > 0):
        return None
    return lower.log().lower()


def _g_upper_on_t_interval(
    t_interval: "arb",
    denominator: Sequence[tuple[Poly, int]],
    numerator: Sequence[tuple[Poly, Fraction]],
) -> "arb | None":
    x = _chi_interval(t_interval)
    total = arb(0)
    for poly, q in numerator:
        if q:
            total += _arb_fraction(q) * _log_abs_upper(poly, x)
    for poly, multiplicity in denominator:
        lower_log = _log_abs_lower(poly, x)
        if lower_log is None:
            return None
        total -= int(multiplicity) * lower_log
    return total.upper()


def _log_abs_bounds_from_roots(data: dict, a: "arb", b: "arb") -> tuple["arb | None", "arb"]:
    """Return lower/upper bounds for log|R(chi(t))| on ``[a,b]``.

    The curve enclosure uses the global derivative bound
    ``|chi'(t)| <= 6*pi`` for ``t in [0,1]``.
    """

    mid = (a + b) / 2
    half_width = (b - a) / 2
    x_mid = _chi_interval(mid)
    curve_radius = 6 * arb.pi() * half_width
    lower_total = arb(abs(data["leading_coeff_abs"])).log().lower()
    upper_total = arb(abs(data["leading_coeff_abs"])).log().upper()
    for root in data["roots"]:
        center = root.mid()
        root_radius = root.rad()
        distance = abs(x_mid - center)
        lower = distance.lower() - curve_radius - root_radius
        upper = distance.upper() + curve_radius + root_radius
        if not (lower > 0):
            return None, upper_total + upper.log().upper()
        lower_total += lower.log().lower()
        upper_total += upper.log().upper()
    return lower_total, upper_total


def _g_upper_on_panel_from_roots(
    a: "arb",
    b: "arb",
    denominator_data: Sequence[tuple[dict, int]],
    numerator_data: Sequence[tuple[dict, Fraction]],
) -> "arb | None":
    total = arb(0)
    for data, q in numerator_data:
        if q:
            _lower, upper = _log_abs_bounds_from_roots(data, a, b)
            total += _arb_fraction(q) * upper
    for data, multiplicity in denominator_data:
        lower, _upper = _log_abs_bounds_from_roots(data, a, b)
        if lower is None:
            return None
        total -= int(multiplicity) * lower
    return total.upper()


def _nonnegative_upper(x: "arb") -> "arb":
    if x <= 0:
        return arb(0)
    upper = x.upper()
    if upper <= 0:
        return arb(0)
    return upper


def _arb_log_mahler_upper_factor(poly: Poly) -> "arb":
    roots = _poly_roots(poly)
    total = arb(abs(leading_coeff(poly))).log().upper()
    one = acb(1)
    two = acb(2)
    four = acb(4)
    for xi in roots:
        root = (one - four * xi).sqrt()
        for zroot in ((one + root) / two, (one - root) / two):
            radius_upper = abs(zroot).upper()
            if radius_upper <= 1:
                continue
            total += radius_upper.log().upper()
    return total.upper()


def arb_log_mahler_upper(
    denominator: Sequence[tuple[Poly, int]],
    *,
    precision: int = 120,
) -> "arb":
    old_prec = ctx.prec
    ctx.prec = precision
    try:
        total = arb(0)
        for poly, multiplicity in denominator:
            total += int(multiplicity) * _arb_log_mahler_upper_factor(poly)
        return total.upper()
    finally:
        ctx.prec = old_prec


def arb_integral_logplus_upper(
    denominator: Sequence[tuple[Poly, int]],
    numerator: Sequence[tuple[Poly, Fraction]],
    *,
    panels: int = 8192,
    max_depth: int = 8,
    precision: int = 120,
) -> dict:
    """Bound the log-plus integral by interval rectangular sums.

    Each panel is recursively bisected if the denominator cannot be bounded
    away from zero.  This simple method is not tight, but it is a genuine upper
    bound when ``failed_panels`` is empty.
    """

    old_prec = ctx.prec
    ctx.prec = precision
    failed: list[dict] = []
    total = arb(0)
    max_panel_upper = arb(0)
    processed = 0
    root_cache: dict[tuple[int, ...], dict] = {}
    denominator_data = [(_root_data(poly, root_cache), mult) for poly, mult in denominator]
    numerator_data = [(_root_data(poly, root_cache), q) for poly, q in numerator]

    def visit(a_num: int, b_num: int, denom: int, depth: int) -> None:
        nonlocal total, max_panel_upper, processed
        a = arb(a_num) / denom
        b = arb(b_num) / denom
        g_upper = _g_upper_on_panel_from_roots(a, b, denominator_data, numerator_data)
        if g_upper is None:
            if depth < max_depth:
                mid = (a_num + b_num) // 2
                visit(a_num, mid, denom, depth + 1)
                visit(mid, b_num, denom, depth + 1)
                return
            failed.append({"a": str(a), "b": str(b), "reason": "denominator_not_separated"})
            return
        f_upper = _nonnegative_upper(g_upper)
        if f_upper > max_panel_upper:
            max_panel_upper = f_upper
        total += (b - a) * f_upper
        processed += 1

    try:
        denom = panels * (2**max_depth)
        step = 2**max_depth
        for j in range(panels):
            visit(j * step, (j + 1) * step, denom, 0)
        return {
            "integral_upper_arb": str(total.upper()),
            "integral_upper": float(total.upper()),
            "max_panel_integrand_upper": str(max_panel_upper.upper()),
            "panels": panels,
            "processed_panels": processed,
            "failed_panels": failed,
            "max_depth": max_depth,
            "precision": precision,
        }
    finally:
        ctx.prec = old_prec


def interval_certify_candidate(
    candidate: dict,
    *,
    target: str = "0.25443677",
    panels: int = 8192,
    max_depth: int = 8,
    precision: int = 120,
) -> dict:
    """Attempt an Arb interval certificate for an already refined candidate."""

    if not HAVE_FLINT:
        return {
            "certified": False,
            "status": "missing_python_flint",
            "note": "Install python-flint to run the Arb interval checker.",
        }

    denominator = [
        (_poly_from_record(item), int(item.get("multiplicity", 1)))
        for item in candidate.get("denominator", [])
    ]
    numerator = [
        (_poly_from_record(item), _q_fraction(item))
        for item in candidate.get("numerator", [])
        if Fraction(str(item.get("q", 0))) != 0
    ]
    deg_q = sum(degree(poly) * multiplicity for poly, multiplicity in denominator)
    if deg_q <= 0:
        return {"certified": False, "status": "invalid_degree", "degQ": deg_q}

    old_prec = ctx.prec
    ctx.prec = precision
    try:
        logm_upper = arb_log_mahler_upper(denominator, precision=precision)
        integral = arb_integral_logplus_upper(
            denominator,
            numerator,
            panels=panels,
            max_depth=max_depth,
            precision=precision,
        )
        if integral["failed_panels"]:
            return {
                "certified": False,
                "status": "failed_integral_panels",
                "degQ": deg_q,
                "logM_upper_arb": str(logm_upper),
                **integral,
            }
        bound_upper = (logm_upper + arb(str(integral["integral_upper_arb"]))) / deg_q
        target_arb = arb(str(target))
        certified = bool(bound_upper < target_arb)
        return {
            "certified": certified,
            "status": "certified_upper_below_target" if certified else "upper_not_below_target",
            "target": target,
            "degQ": deg_q,
            "logM_upper_arb": str(logm_upper),
            "logM_upper": float(logm_upper),
            "bound_upper_arb": str(bound_upper.upper()),
            "bound_upper": float(bound_upper.upper()),
            "bound_exp_upper": math.exp(float(bound_upper.upper())),
            **integral,
        }
    finally:
        ctx.prec = old_prec


def certify_candidate(candidate: dict, *, workdir: str | Path = "runs/certify") -> dict:
    workdir = Path(workdir)
    script_path = export_candidate_for_sage_arb(candidate, workdir / "candidate_certify.sage")
    return {
        "certified": False,
        "status": "not_certified",
        "sage_arb_script": str(script_path),
        "note": (
            "No rigorous interval majoration was computed. "
            "The candidate must remain numeric_only or numeric_refined."
        ),
    }


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "denominator" in item:
                rows.append(item)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSONL file containing candidates")
    parser.add_argument("--top", type=int, default=1)
    parser.add_argument("--target", default="0.25443677")
    parser.add_argument("--panels", type=int, default=8192)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--precision", type=int, default=120)
    parser.add_argument("--out", default="runs/certified.jsonl")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = sorted(_load_jsonl(args.input), key=lambda r: r.get("bound_log", float("inf")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records[: args.top]:
            result = interval_certify_candidate(
                record,
                target=args.target,
                panels=args.panels,
                max_depth=args.max_depth,
                precision=args.precision,
            )
            combined = {"candidate_label": record.get("label"), "candidate_bound_log": record.get("bound_log"), **result}
            f.write(json.dumps(combined, sort_keys=True) + "\n")
            print(
                f"certified={combined['certified']} status={combined['status']} "
                f"bound_upper={combined.get('bound_upper')}"
            )


if __name__ == "__main__":
    main()
