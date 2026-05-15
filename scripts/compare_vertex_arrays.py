#!/usr/bin/env python3
"""Compare two structured vertex arrays and print 0/1 match field as a 2D grid.

Vertex fields have shape [IDim, JDim, Kolor=1] (or [IDim, JDim, 1, K] for K-fields).
For comparison purposes only the K=0 slice is used (or the whole flat array if no K).

By default reads the first two bracketed arrays from one input file.
You can also pass two files and read one array from each file.

For the 26x26 parallelogram grid: --nx 27 --ny 27  (nx+1 vertices per row)
For the 512x512 grid:              --nx 513 --ny 513
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np


def parse_array(bracketed_text: str) -> list[float]:
    s = bracketed_text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace("\n", " ")
    parts = s.split()
    return [float(p) for p in parts if p]


def extract_bracketed_arrays(text: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]", text, re.DOTALL)


def read_text(path: str) -> str:
    try:
        return open(path, "r").read()
    except Exception as exc:
        print(f"Error reading '{path}': {exc}", file=sys.stderr)
        raise


def get_two_arrays(file_a: str, file_b: str | None) -> tuple[list[float], list[float]]:
    if file_b is not None:
        arrays_a = extract_bracketed_arrays(read_text(file_a))
        arrays_b = extract_bracketed_arrays(read_text(file_b))
        if not arrays_a:
            raise ValueError(f"No bracketed array found in '{file_a}'.")
        if not arrays_b:
            raise ValueError(f"No bracketed array found in '{file_b}'.")
        return parse_array(arrays_a[0]), parse_array(arrays_b[0])

    arrays = extract_bracketed_arrays(read_text(file_a))
    if len(arrays) < 2:
        raise ValueError("Could not find two bracketed arrays in input file.")
    return parse_array(arrays[0]), parse_array(arrays[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two vertex arrays and print element-wise match field (1=match, 0=mismatch) "
            "as a 2D grid of shape [IDim x JDim]."
        )
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="out_gtfn.txt",
        help="Input file containing two bracketed arrays (default: out_gtfn.txt)",
    )
    parser.add_argument(
        "--file-b",
        help="Optional second file. If provided, read first array from each file.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for comparison (default: 1e-5).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for comparison (default: 1e-5).",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=27,
        help="IDim size = nx_parallelogram + 1 (default: 27 for 26x26 grid).",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=27,
        help="JDim size = ny_parallelogram + 1 (default: 27 for 26x26 grid).",
    )
    parser.add_argument(
        "--k-index",
        type=int,
        default=0,
        help="Which K-level to compare for K-fields (default: 0).",
    )
    args = parser.parse_args()

    try:
        a_list, b_list = get_two_arrays(args.file, args.file_b)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    a = np.array(a_list)
    b = np.array(b_list)

    n_vertex = args.nx * args.ny  # total vertices (1 kolor)

    # If the array has more elements than nx*ny, assume it has a K dimension.
    # Shape is [IDim, JDim, 1, K] flattened, so total = nx * ny * 1 * K.
    if len(a) > n_vertex:
        k_levels = len(a) // n_vertex
        if k_levels > 1:
            # Reshape to [nx, ny, 1, K] and pick the requested K slice.
            a = a.reshape(args.nx, args.ny, 1, k_levels)[:, :, 0, args.k_index].flatten()
            b = b.reshape(args.nx, args.ny, 1, k_levels)[:, :, 0, args.k_index].flatten()
    else:
        # No K dimension: shape is [IDim, JDim, 1] flattened = nx*ny elements.
        a = a[:n_vertex]
        b = b[:n_vertex]

    n = n_vertex
    matches = [
        1 if abs(float(a[i]) - float(b[i])) <= (args.atol + args.rtol * abs(float(b[i]))) else 0
        for i in range(n)
    ]
    true_count = sum(matches)
    false_count = n - true_count
    mismatch_indices = [i for i, v in enumerate(matches) if v == 0]

    print(f"Vertex grid: {args.nx} x {args.ny}  (total {n} vertices)")
    print(f"Matches: {true_count}/{n} ({true_count / n:.2%})")
    if false_count:
        print(f"Mismatches at flat indices: {mismatch_indices[:20]}{'...' if len(mismatch_indices) > 20 else ''}")
    else:
        print("No mismatches.")

    # Print 2D match grid (IDim as rows, JDim as columns)
    grid = np.array(matches, dtype=int).reshape(args.nx, args.ny)
    print(f"\nMatch grid [IDim=0..{args.nx-1} rows, JDim=0..{args.ny-1} cols]  (1=match, 0=mismatch):")
    for i, row in enumerate(grid):
        print(f"  i={i:3d}: " + " ".join(str(v) for v in row))


if __name__ == "__main__":
    main()
