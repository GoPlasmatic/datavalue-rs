#!/usr/bin/env python3
"""Run the cross-library JSON benchmark and rewrite BENCHMARKS.md tables.

The benchmark output (or a saved log via --from-log) is parsed for the
median time / throughput of every (workload, fixture, library) cell, and
each table inside the BENCH:<id>:BEGIN/END marker pairs in BENCHMARKS.md
is rewritten in place. Narrative around the tables is left untouched.

Usage:
    scripts/update_benchmarks.py                # full criterion run (~10 min)
    scripts/update_benchmarks.py --quick        # short run via criterion --quick
    scripts/update_benchmarks.py --from-log P   # skip bench, parse the log at P
    scripts/update_benchmarks.py --save-log P   # also write raw output to P

Requires only the Python stdlib.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_FILE = REPO_ROOT / "BENCHMARKS.md"

CARGO_BASE = ["cargo", "bench", "--bench", "compare", "--features", "serde_json"]

# Canonical row order per workload — keeps tables stable across runs even if
# criterion shuffles bench order in its output.
# Recommendation split: arena `DataValue` is the recommended type for parse,
# serialize, and access (read-only workloads); `OwnedDataValue` is the
# recommended type for mutation. Each table surfaces only the recommended
# datavalue-rs variant — they don't both appear in the same group.
#
# Parse-side row uses the reused-arena measurement. The fresh-arena bench
# still runs (benches/compare.rs) but isn't surfaced — on these fixture
# sizes the two are within noise.
PARSE_LIBS = [
    "datavalue (reused arena)",
    "serde_json::Value",
    "simd_json (borrowed)",
    "simd_json (owned)",
    "sonic_rs::Value",
    "json-rust",
]
PARSE_DISPLAY = {"datavalue (reused arena)": "datavalue"}
SERIALIZE_LIBS = [
    "datavalue",
    "serde_json::Value",
    "simd_json (owned)",
    "sonic_rs::Value",
    "json-rust",
]
ACCESS_LIBS = SERIALIZE_LIBS
MUTATE_LIBS = [
    "OwnedDataValue (clone + mutate)",
    "serde_json (clone + mutate)",
    "simd_json (clone + mutate)",
    "json-rust (clone + mutate)",
]

FIXTURES = ["twitter", "citm", "canada"]

PARSE_HEADERS = {
    "twitter": "twitter (631 KB)",
    "citm": "citm (1.65 MB)",
    "canada": "canada (2.15 MB)",
}
ACCESS_HEADERS = {
    "twitter": "twitter (statuses[].user, retweet_count)",
    "citm": "citm (events.* id + subTopicIds)",
    "canada": "canada (features[].coordinates[])",
}

# ---- 1. Run cargo bench --------------------------------------------------

def run_bench(quick: bool) -> str:
    cmd = list(CARGO_BASE)
    if quick:
        cmd += ["--", "--quick"]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sys.stderr.write(proc.stdout[-2000:])
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    return proc.stdout

# ---- 2. Parse criterion output ------------------------------------------

NAME_RE = re.compile(r"^(parse|serialize|access|mutate)/([^/]+)/(.+?)(\s+time:.*)?$")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")


def _median(bracket_body: str) -> str | None:
    # bracket_body is like "497.77 µs 498.10 µs 498.55 µs"
    parts = bracket_body.split()
    if len(parts) >= 6:
        return f"{parts[2]} {parts[3]}"
    return None


def parse_log(text: str) -> dict[tuple[str, str, str], dict[str, str]]:
    results: dict[tuple[str, str, str], dict[str, str]] = {}
    current: tuple[str, str, str] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("Benchmarking"):
            continue
        stripped = line.strip()

        if stripped.startswith(("time:", "thrpt:")):
            if current is None:
                continue
            kind = "time" if stripped.startswith("time:") else "thrpt"
            m = BRACKET_RE.search(stripped)
            if not m:
                continue
            median = _median(m.group(1))
            if median:
                results.setdefault(current, {})[kind] = median
            continue

        m = NAME_RE.match(line)
        if not m:
            continue
        group, fixture, lib = m.group(1), m.group(2), m.group(3).strip()
        current = (group, fixture, lib)
        results.setdefault(current, {})
        # Some criterion outputs put name + time on the same line.
        if m.group(4):
            bm = BRACKET_RE.search(m.group(4))
            if bm:
                median = _median(bm.group(1))
                if median:
                    results[current]["time"] = median
    return results

# ---- 3. Unit conversion (for picking the best per column) ---------------

TIME_NS = {"ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}
SIZE_BPS = {
    "B/s": 1.0,
    "KiB/s": 1024.0,
    "MiB/s": 1024.0**2,
    "GiB/s": 1024.0**3,
    "TiB/s": 1024.0**4,
}


def _to_float(s: str | None, table: dict[str, float]) -> float | None:
    if not s:
        return None
    parts = s.split()
    if len(parts) != 2 or parts[1] not in table:
        return None
    try:
        return float(parts[0]) * table[parts[1]]
    except ValueError:
        return None


def time_ns(s):
    return _to_float(s, TIME_NS)


def thrpt_bps(s):
    return _to_float(s, SIZE_BPS)

# ---- 4. Table generation ------------------------------------------------

def _best(results, group, fixture, libs, key, *, lower_is_better):
    best_lib = None
    best_val = float("inf") if lower_is_better else -1.0
    converter = time_ns if key == "time" else thrpt_bps
    for lib in libs:
        v = converter(results.get((group, fixture, lib), {}).get(key))
        if v is None:
            continue
        if (lower_is_better and v < best_val) or (not lower_is_better and v > best_val):
            best_val = v
            best_lib = lib
    return best_lib


def _row(label: str, cells: list[str], any_best: bool) -> str:
    label_md = f"**{label}**" if any_best else label
    return "| " + " | ".join([label_md] + cells) + " |"


def _cell_pair(time_s, thrpt_s, mark_best: bool) -> str:
    if not time_s:
        return "—"
    body = f"{time_s} · {thrpt_s}" if thrpt_s else time_s
    return f"**{body}**" if mark_best else body


def _cell_time(time_s, mark_best: bool) -> str:
    if not time_s:
        return "—"
    return f"**{time_s}**" if mark_best else time_s


def build_parse_table(results) -> str:
    headers = [PARSE_HEADERS[f] for f in FIXTURES]
    lines = ["| Library | " + " | ".join(headers) + " |", "|---|" + "---|" * len(FIXTURES)]
    bests = {f: _best(results, "parse", f, PARSE_LIBS, "thrpt", lower_is_better=False) for f in FIXTURES}
    for lib in PARSE_LIBS:
        cells = []
        for f in FIXTURES:
            r = results.get(("parse", f, lib), {})
            cells.append(_cell_pair(r.get("time"), r.get("thrpt"), bests[f] == lib))
        display = PARSE_DISPLAY.get(lib, lib)
        lines.append(_row(display, cells, any(bests[f] == lib for f in FIXTURES)))
    return "\n".join(lines)


def build_serialize_table(results) -> str:
    lines = [
        "| Library | " + " | ".join(FIXTURES) + " |",
        "|---|" + "---|" * len(FIXTURES),
    ]
    bests = {
        f: _best(results, "serialize", f, SERIALIZE_LIBS, "thrpt", lower_is_better=False)
        for f in FIXTURES
    }
    for lib in SERIALIZE_LIBS:
        cells = []
        for f in FIXTURES:
            r = results.get(("serialize", f, lib), {})
            cells.append(_cell_pair(r.get("time"), r.get("thrpt"), bests[f] == lib))
        lines.append(_row(lib, cells, any(bests[f] == lib for f in FIXTURES)))
    return "\n".join(lines)


def build_access_table(results) -> str:
    headers = [ACCESS_HEADERS[f] for f in FIXTURES]
    lines = ["| Library | " + " | ".join(headers) + " |", "|---|" + "---|" * len(FIXTURES)]
    bests = {
        f: _best(results, "access", f, ACCESS_LIBS, "time", lower_is_better=True)
        for f in FIXTURES
    }
    for lib in ACCESS_LIBS:
        cells = []
        for f in FIXTURES:
            r = results.get(("access", f, lib), {})
            cells.append(_cell_time(r.get("time"), bests[f] == lib))
        lines.append(_row(lib, cells, any(bests[f] == lib for f in FIXTURES)))
    return "\n".join(lines)


def build_mutate_table(results) -> str:
    lines = ["| Library | time | thrpt |", "|---|---|---|"]
    best_t = _best(results, "mutate", "twitter", MUTATE_LIBS, "time", lower_is_better=True)
    best_h = _best(results, "mutate", "twitter", MUTATE_LIBS, "thrpt", lower_is_better=False)
    for lib in MUTATE_LIBS:
        r = results.get(("mutate", "twitter", lib), {})
        time_cell = _cell_time(r.get("time"), best_t == lib)
        thrpt_raw = r.get("thrpt")
        thrpt_cell = "—" if not thrpt_raw else (
            f"**{thrpt_raw}**" if best_h == lib else thrpt_raw
        )
        any_best = best_t == lib or best_h == lib
        lines.append(_row(lib, [time_cell, thrpt_cell], any_best))
    return "\n".join(lines)

# ---- 5. Marker replacement ---------------------------------------------

def replace_block(text: str, marker_id: str, new_body: str) -> str:
    pattern = re.compile(
        rf"(<!-- BENCH:{re.escape(marker_id)}:BEGIN -->\n).*?(\n<!-- BENCH:{re.escape(marker_id)}:END -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        sys.exit(
            f"marker BENCH:{marker_id} not found in {BENCH_FILE.relative_to(REPO_ROOT)}"
        )
    return pattern.sub(lambda m: m.group(1) + new_body + m.group(2), text)

# ---- 6. CLI -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="pass --quick to criterion")
    ap.add_argument("--from-log", metavar="PATH", help="parse this log instead of running cargo bench")
    ap.add_argument("--save-log", metavar="PATH", help="also write raw bench output to this path")
    args = ap.parse_args()

    if args.from_log:
        raw = Path(args.from_log).read_text()
    else:
        raw = run_bench(args.quick)
        if args.save_log:
            Path(args.save_log).write_text(raw)

    results = parse_log(raw)
    if not results:
        sys.exit("no benchmark results parsed — empty or unrecognized output")

    md = BENCH_FILE.read_text()
    md = replace_block(md, "parse", build_parse_table(results))
    md = replace_block(md, "serialize", build_serialize_table(results))
    md = replace_block(md, "access", build_access_table(results))
    md = replace_block(md, "mutate", build_mutate_table(results))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    md = replace_block(md, "updated", f"_Last updated: {stamp} (auto-generated by `scripts/update_benchmarks.py`)._")
    BENCH_FILE.write_text(md)
    print(
        f"updated {BENCH_FILE.relative_to(REPO_ROOT)} — {len(results)} bench results parsed",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
