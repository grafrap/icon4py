#!/usr/bin/env python3
"""Find the first as_fieldop that receives a Kolor:0:0 or empty Kolor domain in infer_domain trace."""
import re, sys

def extract_kolor(domain_str):
    m = re.search(r"'Kolor'.*?SymbolicRange\(start=.*?value=(['\"]?)(-?\d+)['\"]?.*?stop=.*?value=(['\"]?)(-?\d+)['\"]?", domain_str)
    if m:
        return int(m.group(2)), int(m.group(4))
    return None

file_path = sys.argv[1] if len(sys.argv) > 1 else "test_out.txt"
lines = open(file_path).readlines()

# Collect INFER_AS_FIELDOP pairs (before/after)
i = 0
entry_count = 0
while i < len(lines):
    line = lines[i].strip()
    if "[INFER_AS_FIELDOP]" in line and "target_domain_before=" in line:
        before_kolor = extract_kolor(line)
        after_line = lines[i+1].strip() if i+1 < len(lines) else ""
        after_kolor = extract_kolor(after_line)
        entry_count += 1
        # Flag empty Kolor (start==stop) or Kolor width 0
        empty = (after_kolor and after_kolor[0] >= after_kolor[1])
        if empty:
            print(f"=== Entry #{entry_count}: EMPTY/zero Kolor after infer ===")
            print(f"  BEFORE: Kolor={before_kolor}")
            print(f"  AFTER:  Kolor={after_kolor}")
            print(f"  BEFORE line: {line[:200]}")
            print(f"  AFTER  line: {after_line[:200]}")
            print()
    elif "[INFER_AS_FIELDOP]" in line and "target_domain_after=" in line:
        after_kolor = extract_kolor(line)
        if after_kolor and after_kolor[0] >= after_kolor[1]:
            entry_count += 1
            print(f"=== Entry #{entry_count}: EMPTY Kolor (after only) ===")
            print(f"  Kolor={after_kolor}")
            print(f"  line: {line[:200]}")
    i += 1

print(f"\nTotal INFER_AS_FIELDOP entries processed: {entry_count}")
