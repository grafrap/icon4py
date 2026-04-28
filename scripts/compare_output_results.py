#!/usr/bin/env python3
"""
Compare the 'Global mean of' sections and timer reports from two output files.
Extracts lines from 'Global mean of' through the timer section and compares.
"""

import sys
from pathlib import Path


def extract_results_section(filepath):
    """
    Extract the 'Global mean of' section and timer report from a file.
    Returns a tuple of (global_mean_lines, timer_lines).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the start of "Global mean of"
    global_mean_start = None
    for i, line in enumerate(lines):
        if line.startswith('Global mean of'):
            global_mean_start = i
            break
    
    if global_mean_start is None:
        print(f"ERROR: Could not find 'Global mean of' section in {filepath}")
        return None, None
    
    # Find the start of timer report
    timer_start = None
    for i in range(global_mean_start, len(lines)):
        if '===== ICON4Py timer report =====' in lines[i]:
            timer_start = i
            break
    
    if timer_start is None:
        print(f"ERROR: Could not find timer report section in {filepath}")
        return None, None
    
    # Extract both sections
    global_mean_section = lines[global_mean_start:timer_start]
    # timer_section = lines[timer_start:]
    
    return global_mean_section#, timer_section


def compare_sections(section1, section2, section_name):
    """Compare two sections and return True if identical."""
    if section1 == section2:
        print(f"✓ {section_name}: IDENTICAL")
        return True
    else:
        print(f"✗ {section_name}: DIFFERENT")
        # Show first difference
        for i, (line1, line2) in enumerate(zip(section1, section2)):
            if line1 != line2:
                print(f"  First difference at line {i+1}:")
                print(f"    File 1: {line1.rstrip()}")
                print(f"    File 2: {line2.rstrip()}")
                break
        
        # Check if one is longer
        if len(section1) != len(section2):
            print(f"  Length mismatch: File 1 has {len(section1)} lines, File 2 has {len(section2)} lines")
        
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_output_results.py <file1> <file2>")
        print("Example: python compare_output_results.py main_out_unstr.txt main_out.txt")
        sys.exit(1)
    
    file1_path = Path(sys.argv[1])
    file2_path = Path(sys.argv[2])
    
    # Check files exist
    if not file1_path.exists():
        print(f"ERROR: File not found: {file1_path}")
        sys.exit(1)
    
    if not file2_path.exists():
        print(f"ERROR: File not found: {file2_path}")
        sys.exit(1)
    
    print(f"Comparing results from:")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}")
    print()
    
    # Extract sections from both files
    global_mean1 = extract_results_section(file1_path)
    global_mean2 = extract_results_section(file2_path)
    
    if global_mean1 is None or global_mean2 is None:
        sys.exit(1)
    
    # Compare sections
    print("Comparison Results:")
    print("-" * 60)
    
    global_mean_match = compare_sections(global_mean1, global_mean2, "Global mean section")
    # timer_match = compare_sections(timer1, timer2, "Timer section")
    
    print("-" * 60)
    
    if global_mean_match: #and timer_match:
        print("\n✓ All sections match! The output is identical.")
        sys.exit(0)
    else:
        print("\n✗ Outputs differ.")
        sys.exit(1)


if __name__ == '__main__':
    main()
