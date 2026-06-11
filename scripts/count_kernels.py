#!/usr/bin/env python3
"""Count CUDA kernels per stencil in a gt4py cache folder.

Usage:
    python3 scripts/count_kernels.py [cache_folder] [output_file]

Defaults:
    cache_folder  .gt4py_cache/cache_edo  (or .gt_cache/cache_edo if the former is absent)
    output_file   output/kernel_count.txt
"""

import re
import sys
from pathlib import Path

_KERNEL_RE = re.compile(r"__launch_bounds__\([^)]+\)\s+(\w+)\(")


def classify(name: str) -> str:
    return "copy" if name.startswith("copy_") else "compute"


def kernels_in_file(cu_path: Path) -> list[tuple[str, str]]:
    """Return list of (kernel_name, kind) for every __global__ kernel in cu_path."""
    kernels = []
    with cu_path.open(errors="replace") as fh:
        for line in fh:
            if line.startswith("__global__"):
                m = _KERNEL_RE.search(line)
                if m:
                    name = m.group(1)
                    kernels.append((name, classify(name)))
    return kernels


def stencil_name_from_dir(d: Path) -> str:
    """Strip the trailing 64-char hex hash from the directory name."""
    name = d.name
    if len(name) > 65 and name[-64:].isalnum() and name[-(65)] == "_":
        return name[: -(65)]
    return name


def main() -> None:
    # --- argument parsing -----------------------------------------------
    cache_root = None
    output_path = None

    for arg in sys.argv[1:]:
        p = Path(arg)
        if output_path is None and p.suffix == ".txt":
            output_path = p
        elif cache_root is None:
            cache_root = p
        else:
            output_path = p

    if cache_root is None:
        for candidate in (
            Path(".gt4py_cache/cache_edo"),
            Path(".gt_cache/cache_edo"),
        ):
            if candidate.is_dir():
                cache_root = candidate
                break
        if cache_root is None:
            sys.exit("No cache folder found; pass one as first argument.")

    if output_path is None:
        output_path = Path("output/kernel_count.txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- scan -------------------------------------------------------
    results: list[tuple[str, list[tuple[str, str]]]] = []

    for stencil_dir in sorted(cache_root.iterdir()):
        if not stencil_dir.is_dir():
            continue
        cuda_files = sorted((stencil_dir / "src" / "cuda").glob("*.cu"))
        if not cuda_files:
            continue
        cu = cuda_files[0]
        kernels = kernels_in_file(cu)
        if not kernels:
            continue
        results.append((stencil_name_from_dir(stencil_dir), kernels))

    # --- format output -----------------------------------------------
    lines: list[str] = []
    lines.append(f"Cache folder : {cache_root.resolve()}")
    lines.append(f"Stencils found: {len(results)}")
    lines.append("")

    total_compute = total_copy = 0

    for stencil, kernels in results:
        compute = [n for n, k in kernels if k == "compute"]
        copy    = [n for n, k in kernels if k == "copy"]
        total_compute += len(compute)
        total_copy    += len(copy)

        lines.append(f"{'='*70}")
        lines.append(f"Stencil : {stencil}")
        lines.append(f"  Total kernels : {len(kernels)}  (compute={len(compute)}, copy={len(copy)})")
        lines.append(f"  Compute kernels ({len(compute)}):")
        for n in compute:
            lines.append(f"    {n}")
        lines.append(f"  Copy kernels ({len(copy)}):")
        for n in copy:
            lines.append(f"    {n}")
        lines.append("")

    lines.append(f"{'='*70}")
    lines.append(f"TOTALS across all stencils: compute={total_compute}, copy={total_copy}, total={total_compute+total_copy}")

    output = "\n".join(lines) + "\n"
    output_path.write_text(output)
    print(output)
    print(f"[written to {output_path}]")


if __name__ == "__main__":
    main()
