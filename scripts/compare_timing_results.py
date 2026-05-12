#!/usr/bin/env python3
"""
Compare timing results from two benchmark runs.

Reads timing_results.txt from two folders, computes if needed,
and compares median values.
"""

import subprocess
import sys
from pathlib import Path


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
            print(f"  ✓ Generated timing_results.txt")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to generate timing_results.txt: {e}")
            sys.exit(1)
    
    return results_file


def parse_timing_results(file_path: Path) -> dict[str, float]:
    """
    Parse timing_results.txt and extract median values.
    
    Returns dict mapping stencil_name -> median_value
    """
    content = file_path.read_text()
    results = {}
    
    # Skip header lines
    lines = content.split('\n')
    in_data = False
    
    for line in lines:
        # Skip until we hit the separator after header
        if '=' in line and not in_data:
            in_data = True
            continue
        
        if not in_data or not line.strip():
            continue
        
        # Parse data line
        # Format: stencil_name  min  max  mean  stddev  median  iterations
        parts = line.split()
        if len(parts) >= 7:  # At least name + 5 numbers + iterations
            try:
                stencil_name = parts[0]
                median_value = float(parts[5])  # median is the 6th column (index 5)
                results[stencil_name] = median_value
            except (ValueError, IndexError):
                # Skip lines that don't match the expected format
                continue
    
    return results


def compare_results(
    results1: dict[str, float],
    results2: dict[str, float],
    name1: str,
    name2: str,
) -> list[str]:
    """
    Compare two sets of timing results.
    
    Returns formatted comparison lines.
    rel_ratio > 1 means name2 is slower (regression).
    rel_ratio < 1 means name2 is faster (improvement).
    """
    comparison = []
    
    # Header
    header = (
        f"{'Stencil':<56s} "
        f"{name1:<12s} {name2:<12s} "
        f"{'Abs Diff':<12s} {'Ratio (v2/v1)':<12s}"
    )
    comparison.append('=' * 110)
    comparison.append(header)
    comparison.append('=' * 110)
    
    # Get all stencil names (union of both)
    all_stencils = sorted(set(results1.keys()) | set(results2.keys()))
    
    for stencil in all_stencils:
        median1 = results1.get(stencil)
        median2 = results2.get(stencil)
        
        if median1 is None or median2 is None:
            # Skip stencils that don't exist in one of the results
            continue
        
        abs_diff = median2 - median1
        # Ratio: if > 1, name2 is slower; if < 1, name2 is faster
        rel_ratio = (median2 / median1) if median1 != 0 else 0.0
        
        line = (
            f"{stencil:<56s} "
            f"{median1:<12.5f} {median2:<12.5f} "
            f"{abs_diff:<12.5f} {rel_ratio:<12.4f}"
        )
        comparison.append(line)
    
    comparison.append('=' * 110)
    
    return comparison


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python compare_timing_results.py <folder1> <folder2>")
        print("\nCompares timing results from two benchmark runs.")
        print("Generates timing_results.txt if not present.")
        print("\nOutput columns:")
        print("  Median1, Median2: Median exec times (seconds)")
        print("  Abs Diff: median2 - median1 (negative = faster)")
        print("  Rel % Change: % difference (positive = folder1 faster)")
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
    
    # Ensure timing results exist
    print(f"\nChecking {folder1.name}...")
    results_file1 = ensure_timing_results(folder1)
    
    print(f"Checking {folder2.name}...")
    results_file2 = ensure_timing_results(folder2)
    
    # Parse results
    print(f"\nParsing results...")
    results1 = parse_timing_results(results_file1)
    results2 = parse_timing_results(results_file2)
    
    print(f"  {folder1.name}: {len(results1)} stencils")
    print(f"  {folder2.name}: {len(results2)} stencils")
    
    # Compare
    comparison = compare_results(results1, results2, folder1.name, folder2.name)
    
    # Create output folder
    output_folder = Path('output/comparison')
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    name1 = folder1.name
    name2 = folder2.name
    output_file = output_folder / f'{name1}_vs_{name2}.txt'
    
    # Write comparison
    with open(output_file, 'w') as f:
        for line in comparison:
            f.write(line + '\n')
    
    print(f"\n✓ Comparison written to {output_file}")
    print("\nComparison results (first 15 lines):")
    for line in comparison[:15]:
        print(line)


if __name__ == '__main__':
    main()
