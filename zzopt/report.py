"""Report top JSONL candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fmt_q(record: dict) -> str:
    parts = []
    for item in record.get("numerator", []):
        q = item.get("q_rational") or item.get("q")
        parts.append(f"{item['name']}:{q}")
    return ", ".join(parts) or "-"


def _fmt_den(record: dict) -> str:
    return " * ".join(f"{item['name']}^{item.get('multiplicity', 1)}" for item in record.get("denominator", []))


def render_markdown(records: Sequence[dict], *, top: int = 20) -> str:
    lines = ["# Zhang-Zagier Numeric Search Report", ""]
    lines.append("| # | status | bound_log | exp(bound) | degQ | degA | logM | integral | gcd_ok | Q | q |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for idx, rec in enumerate(records[:top], 1):
        lines.append(
            "| "
            f"{idx} | {rec.get('status', '')} | {rec.get('bound_log', math.inf):.10f} | "
            f"{rec.get('bound_exp', math.exp(rec['bound_log']) if 'bound_log' in rec else math.inf):.9f} | "
            f"{rec.get('degQ', '')} | {rec.get('degA', 0):.6f} | "
            f"{rec.get('logM', 0):.10f} | {rec.get('integral', 0):.10f} | "
            f"{rec.get('gcd_ok', '')} | `{_fmt_den(rec)}` | `{_fmt_q(rec)}` |"
        )
    lines.append("")
    lines.append(
        "All rows are numerical unless `status` is explicitly `certified`; "
        "`numeric_only` and `numeric_refined` are not proofs."
    )
    return "\n".join(lines)


def print_records(records: Sequence[dict], *, top: int = 20) -> None:
    for idx, rec in enumerate(records[:top], 1):
        print(f"{idx:02d}. bound_log={rec['bound_log']:.10f} exp={rec.get('bound_exp', math.exp(rec['bound_log'])):.9f}")
        print(f"    status={rec.get('status')} degQ={rec.get('degQ')} degA={rec.get('degA')} gcd_ok={rec.get('gcd_ok', '')}")
        print(f"    logM={rec.get('logM'):.10f} integral={rec.get('integral'):.10f}")
        print(f"    Q={_fmt_den(rec)}")
        print(f"    q={_fmt_q(rec)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="runs/refined.jsonl")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--md-out", default="runs/report.md")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = [r for r in load_jsonl(args.input) if math.isfinite(r.get("bound_log", math.inf))]
    records.sort(key=lambda r: r["bound_log"])
    print_records(records, top=args.top)
    md = render_markdown(records, top=args.top)
    out = Path(args.md_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

