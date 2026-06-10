#!/usr/bin/env python3
"""
Compare timing results from two benchmark runs.

Reads timing_results.txt from two folders, computes if needed,
and compares median values.

Prefers the GT4Py Timer Report (benchmark-framework medians) when both
folders contain a valid report (present and not all-1.0 placeholders).
Falls back to the manually-instrumented [timing] exec values otherwise.
"""

import subprocess
import sys
from pathlib import Path


def parse_gt4py_timer_from_folder(folder: Path) -> "dict[str, float]":
    """
    Scan all .txt files in folder for GT4Py Timer Report sections.

    Returns dict mapping benchmark_name -> median_value.
    Returns empty dict when no valid report is found.
    """
    results = {}
    for txt_file in sorted(folder.glob('*.txt')):
        if txt_file.name == 'timing_results.txt':
            continue
        try:
            content = txt_file.read_text()
        except OSError:
            continue

        lines = content.split('\n')
        in_report = False
        past_header = False

        for line in lines:
            if 'GT4Py Timer Report' in line:
                in_report = True
                past_header = False
                continue

            if not in_report:
                continue

            stripped = line.strip()

            # Separator lines are all dashes (possibly with spaces)
            if stripped and all(c == '-' for c in stripped):
                if not past_header:
                    past_header = True  # first separator = data starts next
                else:
                    break  # second separator = end of report
                continue

            # Skip the column header line
            if 'Benchmark Name' in line:
                continue

            # Data line: "Name | mean | median | stddev | runs"
            if past_header and '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    try:
                        name = parts[0].strip()
                        median_val = float(parts[2])
                        if name:
                            results[name] = median_val
                    except (ValueError, IndexError):
                        pass

    return results


def has_valid_gt4py_timer(results: "dict[str, float]") -> bool:
    """Return True if results is non-empty and not every value is exactly 1.0."""
    if not results:
        return False
    return not all(v == 1.0 for v in results.values())


def ensure_timing_results(folder: Path) -> Path:
    """
    Ensure timing_results.txt exists in folder.

    If not present, runs extract_timing_stats.py to generate it.
    """
    results_file = folder / 'timing_results.txt'
    if not results_file.exists():
        print(f"  timing_results.txt not found in {folder.name}, generating...")
        try:
            subprocess.run(
                ['python3', 'scripts/extract_timing_stats.py', str(folder)],
                check=True,
                capture_output=True
            )
            print(f"  Generated timing_results.txt")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to generate timing_results.txt: {e}")
            sys.exit(1)

    return results_file


def parse_timing_results(file_path: Path) -> "dict[str, float]":
    """
    Parse timing_results.txt and extract median values.

    Returns dict mapping stencil_name -> median_value
    """
    content = file_path.read_text()
    results = {}

    lines = content.split('\n')
    in_data = False

    for line in lines:
        if '=' in line and not in_data:
            in_data = True
            continue

        if not in_data or not line.strip():
            continue

        # Format: stencil_name  min  max  mean  stddev  median  iterations
        parts = line.split()
        if len(parts) >= 7:
            try:
                stencil_name = parts[0]
                median_value = float(parts[5])  # median is column index 5
                results[stencil_name] = median_value
            except (ValueError, IndexError):
                continue

    return results


def compare_results(
    results1: "dict[str, float]",
    results2: "dict[str, float]",
    name1: str,
    name2: str,
) -> "list[str]":
    """
    Compare two sets of timing results.

    Returns formatted comparison lines.
    Ratio < 1 means name2 is faster (improvement).
    Ratio > 1 means name2 is slower (regression).
    """
    all_stencils = sorted(set(results1.keys()) | set(results2.keys()))
    rows = []
    for stencil in all_stencils:
        m1 = results1.get(stencil)
        m2 = results2.get(stencil)
        if m1 is None or m2 is None:
            continue
        # abs_diff = m2 - m1
        ratio = (m2 / m1) if m1 != 0 else 0.0
        rows.append((stencil, m1, m2, ratio))

    # Compute column widths dynamically
    stencil_w = max((len(r[0]) for r in rows), default=10)
    stencil_w = max(stencil_w, len("Stencil"))

    num_decimals = 7
    num_w = max(len(name1), len(name2), num_decimals + 5)  # enough for "0.0001234"
    num_w = max(num_w, 12)

    # diff_w = max(num_w, len("Abs Diff"))
    ratio_label = "Ratio (v2/v1)"
    ratio_w = max(len(ratio_label), 10)

    sep_w = stencil_w + 1 + num_w + 1 + num_w + 1 + 1 + ratio_w

    header = (
        f"{'Stencil':<{stencil_w}s} "
        f"{name1:>{num_w}s} "
        f"{name2:>{num_w}s} "
        # f"{'Abs Diff':>{diff_w}s} "
        f"{ratio_label:>{ratio_w}s}"
    )

    comparison = []
    comparison.append('=' * sep_w)
    comparison.append(header)
    comparison.append('=' * sep_w)

    for stencil, m1, m2, ratio in rows:
        line = (
            f"{stencil:<{stencil_w}s} "
            f"{m1:{num_w}.{num_decimals}f} "
            f"{m2:{num_w}.{num_decimals}f} "
            # f"{abs_diff:{diff_w}.{num_decimals}f} "
            f"{ratio:{ratio_w}.5f}"
        )
        comparison.append(line)

    comparison.append('=' * sep_w)

    return comparison


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python compare_timing_results.py <folder1> <folder2>")
        print("\nCompares timing results from two benchmark runs.")
        print("Generates timing_results.txt if not present.")
        print("\nOutput columns:")
        print("  Median1, Median2: Median exec times (seconds), 7 decimal places")
        # print("  Abs Diff: median2 - median1 (negative = faster)")
        print("  Ratio:    median2 / median1 (< 1 = faster, > 1 = slower)")
        sys.exit(1)

    folder1 = Path(sys.argv[1])
    folder2 = Path(sys.argv[2])

    if not folder1.is_dir():
        print(f"Error: {folder1} is not a directory")
        sys.exit(1)
    if not folder2.is_dir():
        print(f"Error: {folder2} is not a directory")
        sys.exit(1)

    print(f"Comparing {folder1.name} vs {folder2.name}")

    print(f"\nChecking {folder1.name}...")
    results_file1 = ensure_timing_results(folder1)

    print(f"Checking {folder2.name}...")
    results_file2 = ensure_timing_results(folder2)

    print(f"\nChecking for GT4Py Timer Reports...")
    gt4py1 = parse_gt4py_timer_from_folder(folder1)
    gt4py2 = parse_gt4py_timer_from_folder(folder2)
    valid1 = has_valid_gt4py_timer(gt4py1)
    valid2 = has_valid_gt4py_timer(gt4py2)

    if valid1 and valid2:
        print(f"  Both folders have valid GT4Py Timer Reports — using those.")
        results1 = gt4py1
        results2 = gt4py2
        source = "GT4Py Timer (median)"
    else:
        if valid1 and not valid2:
            print(f"  Only {folder1.name} has a valid GT4Py Timer Report — falling back to [timing] exec.")
        elif valid2 and not valid1:
            print(f"  Only {folder2.name} has a valid GT4Py Timer Report — falling back to [timing] exec.")
        else:
            print(f"  No valid GT4Py Timer Reports found — using [timing] exec values.")
        print(f"\nParsing [timing] results...")
        results1 = parse_timing_results(results_file1)
        results2 = parse_timing_results(results_file2)
        source = "[timing] exec (median)"

    print(f"  Source: {source}")
    print(f"  {folder1.name}: {len(results1)} stencils")
    print(f"  {folder2.name}: {len(results2)} stencils")

    comparison = compare_results(results1, results2, folder1.name, folder2.name)

    output_folder = Path('output/comparison')
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / f'{folder1.name}_vs_{folder2.name}.txt'

    with open(output_file, 'w') as f:
        for line in comparison:
            f.write(line + '\n')

    print(f"\nComparison written to {output_file}")
    print()
    for line in comparison:
        print(line)


if __name__ == '__main__':
    main()
