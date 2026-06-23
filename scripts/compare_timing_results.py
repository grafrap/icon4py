#!/usr/bin/env python3
"""
Compare timing results from two benchmark runs.

Reads timing_results.txt from two folders, computes if needed,
and compares median values.

Prefers the GT4Py Timer Report (benchmark-framework medians) when both
folders contain a valid report (present and not all-1.0 placeholders).
Falls back to the manually-instrumented [timing] exec values otherwise.

Columns: Stencil | Entity | Connectivities | folder1 (ms) | folder2 (ms) | Ratio
Ratio < 1 means folder2 is faster. Speedup = folder1/folder2 (>1 = folder2 faster).
"""

import re
import subprocess
import sys
from pathlib import Path


# ── Stencil metadata: entity type and connectivity from test files ─────────────

_TEST_DIRS = [
    Path(__file__).resolve().parent.parent /
        "model/atmosphere/dycore/tests/dycore/stencil_tests",
    Path(__file__).resolve().parent.parent /
        "model/atmosphere/diffusion/tests/diffusion/stencil_tests",
]

_CONN_DIMS = {
    "E2CDim":   "E2C",   "E2VDim":   "E2V",   "E2C2VDim": "E2C2V",
    "E2C2EDim": "E2C2E", "E2C2EODim":"E2C2EO","C2EDim":   "C2E",
    "C2E2CDim": "C2E2C", "C2E2CODim":"C2E2CO","C2VDim":   "C2V",
    "V2EDim":   "V2E",   "V2CDim":   "V2C",
}
_ENTITY_DIMS = {
    "EdgeDim": "Edge", "CellDim": "Cell", "VertexDim": "Vertex",
}

_metadata_cache: "dict[str, tuple[str, str]]" = {}


def _analyze_test(snake_name: str) -> "tuple[str, str]":
    """Return (entity_str, connectivity_str) for a test given its snake_case name."""
    if snake_name in _metadata_cache:
        return _metadata_cache[snake_name]
    for d in _TEST_DIRS:
        f = d / f"{snake_name}.py"
        if not f.exists():
            continue
        txt = f.read_text(errors="replace")
        entities = sorted({e for k, e in _ENTITY_DIMS.items() if k in txt})
        conns    = sorted({c for k, c in _CONN_DIMS.items()  if k in txt})
        result   = ("+".join(entities) or "?", ",".join(conns) or "—")
        _metadata_cache[snake_name] = result
        return result
    _metadata_cache[snake_name] = ("?", "—")
    return ("?", "—")


def _snake_from_output_filename(stem: str) -> str:
    """Extract the snake_case test name from an output file stem."""
    n = re.sub(r'^\d+_', '', stem)
    n = re.sub(r'_\d+_\d+_.*$', '', n)
    n = re.sub(r'_\d+_(structured|unstructured|baseline).*$', '', n, flags=re.I)
    n = re.sub(r'_(structured|unstructured|baseline|retry|fixed|diff_retry|fd).*$',
               '', n, flags=re.I)
    n = re.sub(r'_\d+$', '', n)
    return n


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_gt4py_timer_from_folder(
    folder: Path,
) -> "tuple[dict[str, float], dict[str, tuple[str,str]]]":
    """Scan all .txt files in folder for GT4Py Timer Report sections.

    Returns:
        results   – dict benchmark_name -> median_s
        meta      – dict benchmark_name -> (entity_str, conn_str)
    """
    results: "dict[str, float]" = {}
    meta:    "dict[str, tuple[str,str]]" = {}

    for txt_file in sorted(folder.glob("*.txt")):
        if txt_file.name in ("timing_results.txt", "summary.txt"):
            continue
        try:
            content = txt_file.read_text(errors="replace")
        except OSError:
            continue

        snake = _snake_from_output_filename(txt_file.stem)
        entity, conn = _analyze_test(snake)

        lines = content.split("\n")
        in_report = False
        past_header = False

        for line in lines:
            if "GT4Py Timer Report" in line:
                in_report = True
                past_header = False
                continue
            if not in_report:
                continue
            stripped = line.strip()
            if stripped and all(c == "-" for c in stripped):
                if not past_header:
                    past_header = True
                else:
                    break
                continue
            if "Benchmark Name" in line:
                continue
            if past_header and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        name = parts[0].strip()
                        median_val = float(parts[2])
                        if name:
                            results[name] = median_val
                            meta[name] = (entity, conn)
                    except (ValueError, IndexError):
                        pass

    return results, meta


def has_valid_gt4py_timer(results: "dict[str, float]") -> bool:
    if not results:
        return False
    return not all(v == 1.0 for v in results.values())


def ensure_timing_results(folder: Path) -> Path:
    results_file = folder / "timing_results.txt"
    if not results_file.exists():
        print(f"  timing_results.txt not found in {folder.name}, generating...")
        try:
            subprocess.run(
                ["python3", "scripts/extract_timing_stats.py", str(folder)],
                check=True, capture_output=True,
            )
            print("  Generated timing_results.txt")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to generate timing_results.txt: {e}")
            sys.exit(1)
    return results_file


def parse_timing_results(file_path: Path) -> "dict[str, float]":
    content = file_path.read_text()
    results: "dict[str, float]" = {}
    lines = content.split("\n")
    in_data = False
    for line in lines:
        if "=" in line and not in_data:
            in_data = True
            continue
        if not in_data or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                results[parts[0]] = float(parts[5])
            except (ValueError, IndexError):
                continue
    return results


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_results(
    results1: "dict[str, float]",
    results2: "dict[str, float]",
    meta:     "dict[str, tuple[str,str]]",
    name1: str,
    name2: str,
) -> "list[str]":
    """Build formatted comparison lines.

    Columns: Stencil | Entity | Connectivities | v1 (ms) | v2 (ms) | Speedup (v1/v2)
    Speedup > 1 means folder2 is faster.
    """
    all_stencils = sorted(set(results1.keys()) | set(results2.keys()))
    rows = []
    for stencil in all_stencils:
        m1 = results1.get(stencil)
        m2 = results2.get(stencil)
        if m1 is None or m2 is None:
            continue
        entity, conn = meta.get(stencil, ("?", "—"))
        speedup = m1 / m2 if m2 != 0 else 0.0
        rows.append((stencil, entity, conn, m1, m2, speedup))

    if not rows:
        return ["(no common stencils found)"]

    stencil_w = max(max(len(r[0]) for r in rows), len("Stencil"))
    entity_w  = max(max(len(r[1]) for r in rows), len("Entity"))
    conn_w    = max(max(len(r[2]) for r in rows), len("Connectivities"))
    num_w     = max(len(name1), len(name2), 12)
    spd_label = "Speedup(v1/v2)"
    spd_w     = max(len(spd_label), 14)

    sep_w = stencil_w + entity_w + conn_w + 2 * num_w + spd_w + 10

    header = (
        f"{'Stencil':<{stencil_w}}  "
        f"{'Entity':<{entity_w}}  "
        f"{'Connectivities':<{conn_w}}  "
        f"{name1:>{num_w}}  "
        f"{name2:>{num_w}}  "
        f"{spd_label:>{spd_w}}"
    )

    lines = []
    lines.append("=" * sep_w)
    lines.append(
        f"  {name1} (v1)  vs  {name2} (v2)\n"
        f"  Speedup = v1/v2: >1.0 means v2 is FASTER, <1.0 means v2 is SLOWER\n"
        f"  Timing source: GT4Py Timer median (seconds)"
    )
    lines.append("=" * sep_w)
    lines.append(header)
    lines.append("=" * sep_w)

    for stencil, entity, conn, m1, m2, speedup in rows:
        # Mark fast/slow
        if speedup >= 1.05:
            marker = ">>"   # v2 faster by ≥5%
        elif speedup <= 0.95:
            marker = "<<"   # v2 slower by ≥5%
        else:
            marker = "  "
        line = (
            f"{marker} {stencil:<{stencil_w}}  "
            f"{entity:<{entity_w}}  "
            f"{conn:<{conn_w}}  "
            f"{m1 * 1000:{num_w}.3f}  "
            f"{m2 * 1000:{num_w}.3f}  "
            f"{speedup:{spd_w}.4f}"
        )
        lines.append(line)

    lines.append("=" * sep_w)

    # Summary
    faster = sum(1 for *_, sp in rows if sp >= 1.05)
    slower = sum(1 for *_, sp in rows if sp <= 0.95)
    same   = len(rows) - faster - slower
    lines.append(
        f"\nSummary: {faster} faster (>>), {slower} slower (<<), {same} within 5% — "
        f"{len(rows)} total stencils"
    )
    lines.append(f"Note: timings in milliseconds. Speedup = v1/v2.")

    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_timing_results.py <folder1> <folder2>")
        print("\nCompares timing results from two benchmark runs.")
        print("Generates timing_results.txt if not present.")
        print("\nOutput columns:")
        print("  Entity:         Edge / Cell / Vertex (from test file analysis)")
        print("  Connectivities: E2C, V2E, E2C2EO, ... (neighbour dims used)")
        print("  v1, v2:         Median exec times in milliseconds")
        print("  Speedup:        v1/v2  (>1 = v2 faster, <1 = v2 slower, >> ≥5%, << ≥5%)")
        sys.exit(1)

    folder1 = Path(sys.argv[1])
    folder2 = Path(sys.argv[2])

    for folder in (folder1, folder2):
        if not folder.is_dir():
            print(f"Error: {folder} is not a directory")
            sys.exit(1)

    print(f"Comparing {folder1.name} vs {folder2.name}")

    print(f"\nChecking {folder1.name}...")
    results_file1 = ensure_timing_results(folder1)
    print(f"Checking {folder2.name}...")
    results_file2 = ensure_timing_results(folder2)

    print("\nLooking for GT4Py Timer Reports...")
    gt4py1, meta1 = parse_gt4py_timer_from_folder(folder1)
    gt4py2, meta2 = parse_gt4py_timer_from_folder(folder2)
    valid1, valid2 = has_valid_gt4py_timer(gt4py1), has_valid_gt4py_timer(gt4py2)

    if valid1 and valid2:
        print("  Both folders have valid GT4Py Timer Reports — using those.")
        results1, results2 = gt4py1, gt4py2
        meta = {**meta1, **meta2}   # merge; meta1 wins on collision
        source = "GT4Py Timer (median, ms)"
    else:
        if valid1 and not valid2:
            print(f"  Only {folder1.name} has GT4Py Timer — falling back to [timing] exec.")
        elif valid2 and not valid1:
            print(f"  Only {folder2.name} has GT4Py Timer — falling back to [timing] exec.")
        else:
            print("  No valid GT4Py Timer Reports found — using [timing] exec values.")
        print("\nParsing [timing] results...")
        results1 = parse_timing_results(results_file1)
        results2 = parse_timing_results(results_file2)
        meta = {}
        source = "[timing] exec (median, ms)"

    print(f"  Source: {source}")
    print(f"  {folder1.name}: {len(results1)} stencils")
    print(f"  {folder2.name}: {len(results2)} stencils")

    comparison = compare_results(results1, results2, meta, folder1.name, folder2.name)

    output_folder = Path("output/comparison")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{folder1.name}_vs_{folder2.name}.txt"

    with open(output_file, "w") as f:
        for line in comparison:
            f.write(line + "\n")

    print(f"\nComparison written to {output_file}\n")
    for line in comparison:
        print(line)


if __name__ == "__main__":
    main()
