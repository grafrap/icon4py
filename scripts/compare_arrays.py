#!/usr/bin/env python3
"""
Compare two arrays found in a text file (like `out_gtfn.txt`).
Finds the first two bracketed arrays and compares them element-wise.
Saves/prints a boolean list indicating matches (using math.isclose by default).
"""

import re
import argparse
import math
import json
import sys


def parse_array(bracketed_text: str):
    s = bracketed_text.strip()
    # remove surrounding brackets if present
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    # normalize whitespace and split
    s = s.replace('\n', ' ')
    parts = s.split()
    return [float(p) for p in parts if p]


def main():
    p = argparse.ArgumentParser(description='Compare two arrays in a text file and output boolean matches')
    p.add_argument('file', nargs='?', default='out_gtfn.txt', help='Input file containing two arrays')
    p.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for comparison')
    p.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance for comparison')
    p.add_argument('--out', help='Optional output file (JSON) to write the boolean result list')
    args = p.parse_args()

    try:
        text = open(args.file, 'r').read()
    except Exception as e:
        print(f'Error reading file {args.file}: {e}', file=sys.stderr)
        sys.exit(2)

    # find bracketed arrays (non-greedy)
    groups = re.findall(r"\[[^\]]+\]", text, re.DOTALL)
    if len(groups) < 2:
        print('Could not find two bracketed arrays in the file.', file=sys.stderr)
        sys.exit(3)

    a = parse_array(groups[0])
    b = parse_array(groups[1])

    n = min(len(a), len(b))
    matches = [math.isclose(a[i], b[i], rel_tol=args.rtol, abs_tol=args.atol) for i in range(n)]

    # if lengths differ, mark remaining elements as False
    if len(a) != len(b):
        longer = max(len(a), len(b))
        matches.extend([False] * (longer - n))

    total = len(matches)
    true_count = sum(1 for v in matches if v)
    false_count = total - true_count

    print(f'Matches: {true_count}/{total} ({true_count/total:.2%})')
    if false_count:
        mismatches = [i for i, v in enumerate(matches) if not v]

        print('Mismatches at indices:', mismatches, end='')
    else:
        print('All elements matched within given tolerances.')

    if args.out:
        try:
            with open(args.out, 'w') as f:
                json.dump(matches, f)
            print(f'Wrote boolean array to {args.out}')
        except Exception as e:
            print(f'Error writing output file {args.out}: {e}', file=sys.stderr)
            sys.exit(4)

    # --- User-requested splitting and mapping ---
    # lengths for the three parts
    l1 = 13 * 17
    l2 = 14 * 16
    l3 = 13 * 16
    total_needed = l1 + l2 + l3

    if len(matches) < total_needed:
        # pad unknown comparisons as False
        matches.extend([False] * (total_needed - len(matches)))
    elif len(matches) > total_needed:
        # keep only the first required values
        matches = matches[:total_needed]

    ints = [1 if v else 0 for v in matches]

    p1 = ints[0:l1]
    p2 = ints[l1:l1 + l2]
    p3 = ints[l1 + l2:total_needed]

    def map_part(part, orig_rows, orig_cols, target_rows=17, target_cols=14):
        # build rows from part using orig_cols, pad each row to target_cols,
        # then pad extra rows to reach target_rows
        grid = []
        for r in range(orig_rows):
            start = r * orig_cols
            row = part[start:start + orig_cols]
            if len(row) < orig_cols:
                row = row + [0] * (orig_cols - len(row))
            if target_cols > orig_cols:
                row = row + [0] * (target_cols - orig_cols)
            elif target_cols < orig_cols:
                row = row[:target_cols]
            grid.append(row)
        # pad additional rows if needed
        while len(grid) < target_rows:
            grid.append([0] * target_cols)
        # if too many rows, truncate
        if len(grid) > target_rows:
            grid = grid[:target_rows]
        return grid

    # original shapes for the three parts
    grid1 = map_part(p1, orig_rows=17, orig_cols=13, target_rows=17, target_cols=14)
    grid2 = map_part(p2, orig_rows=16, orig_cols=14, target_rows=17, target_cols=14)
    grid3 = map_part(p3, orig_rows=16, orig_cols=13, target_rows=17, target_cols=14)

    print('\nMapped grids (17x14) — values are 1 for match, 0 for mismatch/unknown:')
    print('\nGrid 1:')
    for row in grid1:
        print(' '.join(str(x) for x in row))
    print('\nGrid 2:')
    for row in grid2:
        print(' '.join(str(x) for x in row))
    print('\nGrid 3:')
    for row in grid3:
        print(' '.join(str(x) for x in row))

    # if JSON output requested, also write grids
    if args.out:
        try:
            with open(args.out + '.grids.json', 'w') as f:
                json.dump({'matches': matches, 'grid1': grid1, 'grid2': grid2, 'grid3': grid3}, f)
            print(f'Wrote grids to {args.out}.grids.json')
        except Exception as e:
            print(f'Error writing grids output file {args.out}.grids.json: {e}', file=sys.stderr)
            sys.exit(5)


if __name__ == '__main__':
    main()
