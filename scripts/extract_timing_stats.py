#!/usr/bin/env python3
"""
Extract timing statistics from output files.

Reads consecutive [timing] lines from each file, calculates min/max/mean/stddev/median,
and appends statistics to both the original file and a summary.txt file.
"""

import re
import sys
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional


def parse_timing_lines(lines: "list[str]") -> "list[float]":
    """
    Extract exec timing values from [timing] lines.
    
    Looks for the 'exec=' value in lines like:
    [timing] name pack=0.01s exec=0.00s unpack=0.00s total=0.02s
    """
    values = []
    for line in lines:
        # Match the 'exec=XXs' pattern
        match = re.search(r'exec=([0-9.]+)s', line)
        if match:
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass
    return values


def group_timing_blocks(file_content: str) -> "list[tuple[str, list[str]]]":
    """
    Group consecutive [timing] lines by their prefix (e.g., stencil name).
    
    Returns a list of (prefix, lines) tuples.
    """
    lines = file_content.split('\n')
    blocks = []
    current_block = []
    current_prefix = None
    
    for line in lines:
        if line.startswith('[timing]'):
            # Extract the prefix (stencil name, typically the second token)
            # Format: [timing] <name> ...
            match = re.match(r'\[timing\]\s+(\S+)\s+', line)
            if match:
                prefix = match.group(1)
                
                # If we're starting a new prefix, save the old block
                if current_prefix is not None and prefix != current_prefix:
                    if current_block:
                        blocks.append((current_prefix, current_block))
                    current_block = []
                
                current_prefix = prefix
                current_block.append(line)
            else:
                # Line doesn't match expected format, still add it
                if current_prefix is not None:
                    current_block.append(line)
        else:
            # Non-timing line encountered
            if current_block and current_prefix:
                blocks.append((current_prefix, current_block))
                current_block = []
                current_prefix = None
    
    # Don't forget the last block
    if current_block and current_prefix:
        blocks.append((current_prefix, current_block))
    
    return blocks


def calculate_stats(values: "list[float]") -> Optional[dict]:
    """
    Calculate statistics for a list of values.
    
    Returns a dict with min, max, mean, stddev, median, and count.
    """
    if not values:
        return None
    
    if len(values) == 1:
        return {
            'min': values[0],
            'max': values[0],
            'mean': values[0],
            'stddev': 0.0,
            'median': values[0],
            'count': 1,
        }
    
    return {
        'min': min(values),
        'max': max(values),
        'mean': mean(values),
        'stddev': stdev(values),
        'median': median(values),
        'count': len(values),
    }


def format_stats(prefix: str, stats: dict) -> "tuple[str, str]":
    """Format statistics as header and data rows.

    Returns (header_row, data_row) tuple.
    """
    w = 13  # column width: 8 decimals + "0." + sign = 11 chars minimum, 13 gives padding
    header = (
        f"{'TIMING STATISTICS':<56s} "
        f"{'Min':<{w}s} {'Max':<{w}s} {'Mean':<{w}s} "
        f"{'StdDev':<{w}s} {'Median':<{w}s} {'Iterations':<4s}"
    )
    data = (
        f"{prefix:<56s} "
        f"{stats['min']:<{w}.8f} {stats['max']:<{w}.8f} "
        f"{stats['mean']:<{w}.8f} {stats['stddev']:<{w}.8f} "
        f"{stats['median']:<{w}.8f} {stats['count']:<4d}"
    )
    return (header, data)


def process_file(file_path: Path) -> "tuple[list[str], list[str]]":
    """
    Process a single file and return the formatted statistics lines.
    
    Also appends statistics to the file itself.
    Skips the first timing run (compilation) for each block.
    
    Returns (headers, data_lines) tuple.
    """
    content = file_path.read_text()
    blocks = group_timing_blocks(content)
    
    headers = []
    data_lines = []
    
    # Process each block
    for i, (prefix, timing_lines) in enumerate(blocks):
        values = parse_timing_lines(timing_lines)
        
        # Skip the first timing run (compilation)
        if len(values) > 1:
            values = values[1:]
        
        stats = calculate_stats(values)
        
        if stats:
            header, data = format_stats(prefix, stats)
            # Only add header once (for the first block)
            if i == 0:
                headers.append(header)
            data_lines.append(data)
    
    # Append statistics to the original file
    if data_lines:
        with open(file_path, 'a') as f:
            f.write('\n\n')
            f.write('=' * 130 + '\n')
            for header in headers:
                f.write(header + '\n')
            f.write('=' * 130 + '\n')
            for line in data_lines:
                f.write(line + '\n')
    
    return (headers, data_lines)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python extract_timing_stats.py <folder>")
        print("\nExtracts timing statistics from [timing] lines in output files.")
        sys.exit(1)
    
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)
    
    # Find all output files (txt files in the directory)
    output_files = sorted(folder.glob('*.txt'))
    
    if not output_files:
        print(f"No .txt files found in {folder}")
        sys.exit(1)
    
    print(f"Processing {len(output_files)} files from {folder}")
    
    # Collect all stats
    all_headers = []
    all_data = []
    total_lines = 0
    
    for file_path in output_files:
        print(f"  Processing {file_path.name}...", end='', flush=True)
        try:
            headers, data_lines = process_file(file_path)
            if headers:
                all_headers = headers  # Use the first file's header format
            all_data.extend(data_lines)
            total_lines += len(data_lines)
            print(f" {len(data_lines)} timing blocks")
        except Exception as e:
            print(f" ERROR: {e}")
    
    # Write summary file
    results_file = folder / 'timing_results.txt'
    print(f"\nWriting results to {results_file}...")
    
    with open(results_file, 'w') as f:
        f.write('=' * 130 + '\n')
        for header in all_headers:
            f.write(header + '\n')
        f.write('=' * 130 + '\n')
        for line in all_data:
            f.write(line + '\n')
    
    print(f"✓ Wrote {total_lines} statistic lines to {results_file}")


if __name__ == '__main__':
    main()
