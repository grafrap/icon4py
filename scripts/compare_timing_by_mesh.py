#!/usr/bin/env python3
"""
Compare timing results from two benchmark runs, broken down by mesh size.

Each .txt file in a benchmark folder is expected to start with a header line:
  === <stencil_name> <mesh_size> ===

Results are presented per (stencil, mesh_size) pair so different grid sizes
are not merged together.

Usage:
  python scripts/compare_timing_by_mesh.py <folder1> <folder2>

Output columns:
  Stencil, Mesh: identifier
  v1 (median), v2 (median): median exec times in seconds
  Abs Diff: v2 - v1 (negative = v2 faster)
  Ratio:    v2 / v1  (< 1 = v2 faster, > 1 = v2 slower)
"""

import re
import sys
from pathlib import Path
from statistics import median, stdev, mean


# ── helpers copied / adapted from extract_timing_stats.py ────────────────────

def _parse_exec_values(file_content: str) -> "list[float]":
    """Extract all exec= values from [timing] lines, skipping the first (JIT warmup)."""
    values = []
    for line in file_content.splitlines():
        m = re.search(r'exec=([0-9.]+)s', line)
        if m:
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
    # skip first (compilation/warmup) run
    return values[1:] if len(values) > 1 else values


def _median(values: "list[float]") -> "float | None":
    if not values:
        return None
    return median(values)


def _extract_header(file_content: str) -> "tuple[str, str] | None":
    """
    Return (stencil_name, mesh_size) from the first === ... === header line.
    Returns None if not found.
    """
    m = re.match(r'===\s+(\S+)\s+(\S+)\s+===', file_content.lstrip())
    if m:
        return m.group(1), m.group(2)
    return None


# ── per-folder scanning ───────────────────────────────────────────────────────

def scan_folder(folder: Path) -> "dict[tuple[str,str], float]":
    """
    Scan all .txt files in folder (except timing_results.txt / summary.txt).

    Returns dict mapping (stencil_name, mesh_size) -> median exec time.
    """
    SKIP = {"timing_results.txt", "summary.txt"}
    results: "dict[tuple[str,str], float]" = {}

    for path in sorted(folder.glob("*.txt")):
        if path.name in SKIP:
            continue
        content = path.read_text()
        header = _extract_header(content)
        if header is None:
            # fall back: try to parse mesh size from filename suffix _NNN.txt
            m = re.search(r'_(\d+)\.txt$', path.name)
            if m:
                # use filename stem prefix as stencil name placeholder
                stem = re.sub(r'_\d+$', '', re.sub(r'^\d+_', '', path.stem))
                header = (stem, m.group(1))
            else:
                print(f"  [skip] {path.name}: no mesh header found", file=sys.stderr)
                continue

        stencil, mesh = header
        values = _parse_exec_values(content)
        med = _median(values)
        if med is None:
            print(f"  [skip] {path.name}: no exec timing lines", file=sys.stderr)
            continue

        key = (stencil, mesh)
        if key in results:
            # multiple files for same (stencil, mesh) — keep the smaller median
            results[key] = min(results[key], med)
        else:
            results[key] = med

    return results


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt(v: "float | None", w: int, decimals: int = 7) -> str:
    if v is None:
        return f"{'N/A':>{w}s}"
    return f"{v:{w}.{decimals}f}"


def build_table(
    results1: "dict[tuple[str,str], float]",
    results2: "dict[tuple[str,str], float]",
    name1: str,
    name2: str,
) -> "list[str]":
    all_keys = sorted(set(results1) | set(results2))

    if not all_keys:
        return ["(no matching results)"]

    # column widths
    stencil_w = max(len(k[0]) for k in all_keys)
    stencil_w = max(stencil_w, len("Stencil"))
    mesh_w = max(len(k[1]) for k in all_keys)
    mesh_w = max(mesh_w, len("Mesh"))
    num_decimals = 7
    num_w = max(len(name1), len(name2), num_decimals + 5, 12)
    diff_w = max(num_w, len("Abs Diff"))
    ratio_label = "Ratio (v2/v1)"
    ratio_w = max(len(ratio_label), 13)

    sep_w = stencil_w + 1 + mesh_w + 1 + num_w + 1 + num_w + 1 + diff_w + 1 + ratio_w

    header = (
        f"{'Stencil':<{stencil_w}s} "
        f"{'Mesh':>{mesh_w}s} "
        f"{name1:>{num_w}s} "
        f"{name2:>{num_w}s} "
        f"{'Abs Diff':>{diff_w}s} "
        f"{ratio_label:>{ratio_w}s}"
    )

    lines = ["=" * sep_w, header, "=" * sep_w]

    prev_stencil = None
    for stencil, mesh in all_keys:
        # blank separator between stencil groups
        if prev_stencil is not None and stencil != prev_stencil:
            lines.append("")
        prev_stencil = stencil

        m1 = results1.get((stencil, mesh))
        m2 = results2.get((stencil, mesh))

        if m1 is not None and m2 is not None:
            abs_diff = m2 - m1
            ratio = (m2 / m1) if m1 != 0 else 0.0
            diff_str = f"{abs_diff:{diff_w}.{num_decimals}f}"
            ratio_str = f"{ratio:{ratio_w}.5f}"
        else:
            abs_diff = None
            ratio = None
            diff_str = f"{'N/A':>{diff_w}s}"
            ratio_str = f"{'N/A':>{ratio_w}s}"

        line = (
            f"{stencil:<{stencil_w}s} "
            f"{mesh:>{mesh_w}s} "
            f"{_fmt(m1, num_w)} "
            f"{_fmt(m2, num_w)} "
            f"{diff_str} "
            f"{ratio_str}"
        )
        lines.append(line)

    lines.append("=" * sep_w)
    return lines


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_timing_by_mesh.py <folder1> <folder2>")
        print()
        print("Compares exec-time medians per (stencil, mesh_size) pair.")
        print("Ratio < 1 means folder2 is faster.")
        sys.exit(1)

    folder1 = Path(sys.argv[1])
    folder2 = Path(sys.argv[2])

    for f in (folder1, folder2):
        if not f.is_dir():
            print(f"Error: {f} is not a directory")
            sys.exit(1)

    print(f"Scanning {folder1.name}...")
    r1 = scan_folder(folder1)
    print(f"  {len(r1)} (stencil, mesh) entries")

    print(f"Scanning {folder2.name}...")
    r2 = scan_folder(folder2)
    print(f"  {len(r2)} (stencil, mesh) entries")

    table = build_table(r1, r2, folder1.name, folder2.name)

    out_dir = Path("output/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{folder1.name}_vs_{folder2.name}_by_mesh.txt"
    with open(out_file, "w") as fh:
        for line in table:
            fh.write(line + "\n")

    print(f"\nComparison written to {out_file}\n")
    for line in table:
        print(line)


if __name__ == "__main__":
    main()
