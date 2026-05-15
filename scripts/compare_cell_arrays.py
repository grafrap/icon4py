#!/usr/bin/env python3
"""Compare two cell arrays and print 0/1 match field split into up/down halves.

By default, this script reads the first two bracketed arrays from one input file.
You can also pass two files and read one array from each file.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable
from gt4py.next.modules.translator import transform_to_unstructured
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
	except Exception as exc:  # pragma: no cover - keeps clear CLI error path
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


def chunked(values: Iterable[int], cols: int) -> list[list[int]]:
	v = list(values)
	return [v[i : i + cols] for i in range(0, len(v), cols)]


def print_section(title: str, values: list[int], cols: int) -> None:
	print(f"\n{title} (len={len(values)}):")
	for row in chunked(values, cols):
		print(" ".join(str(x) for x in row))


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Compare two arrays and print element-wise match field (1=match, 0=mismatch) "
			"split into up-cells (first half) and down-cells (second half)."
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
		"--split-index",
		type=int,
		help="Index where array is split into up/down halves (default: nx*ny).",
	)
	parser.add_argument(
		"--nx",
		type=int,
		default=26,
		help="Number of cells in x direction (default: 26).",
	)
	parser.add_argument(
		"--ny",
		type=int,
		default=26,
		help="Number of cells in y direction (default: 26).",
	)
	parser.add_argument(
		"--cols",
		type=int,
		default=26,
		help="Values printed per line for each half (default: 26).",
	)
	args = parser.parse_args()

	try:
		a_, b_ = get_two_arrays(args.file, args.file_b)
		trafo = transform_to_unstructured(np.array(a_), args.nx, "Cell")[1]
		a = np.array(a_)[trafo] # only necessary, if we do unstructured backend !!!
		b = np.array(b_)[trafo]
	except Exception as exc:
		print(str(exc), file=sys.stderr)
		sys.exit(2)

	n = min(len(a), len(b))
	if len(a) != len(b):
		print(
			f"Warning: different lengths (a={len(a)}, b={len(b)}). "
			f"Using first {n} values.",
			file=sys.stderr,
		)

	expected_n = 2 * args.nx * args.ny
	if n < expected_n:
		print(
			f"Error: not enough values for cell layout 2*nx*ny={expected_n} (got {n}).",
			file=sys.stderr,
		)
		sys.exit(2)
	if n > expected_n:
		print(
			f"Warning: array has {n} values, but cell layout expects {expected_n}. "
			"Ignoring trailing values.",
			file=sys.stderr,
		)
		n = expected_n
		a = a[:n]
		b = b[:n]

	matches = [1 if abs(a[i] - b[i]) <= (args.atol + args.rtol * abs(b[i])) else 0 for i in range(n)]
	true_count = sum(matches)
	false_count = n - true_count
	mismatch_indices = [i for i, v in enumerate(matches) if v == 0]

	split = args.split_index if args.split_index is not None else (args.nx * args.ny)
	split = max(0, min(split, n))

	up = matches[:split]
	down = matches[split:]

	print(f"Compared length: {n}")
	print(f"Split index: {split}")
	print(f"Matches: {true_count}/{n} ({true_count / n:.2%})")
	if false_count:
		print(f"Mismatches at indices: {mismatch_indices}")
	else:
		print("No mismatches.")

	print_section("Up-cells match field", up, args.cols)
	print_section("Down-cells match field", down, args.cols)


if __name__ == "__main__":
	main()
