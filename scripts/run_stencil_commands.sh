#!/usr/bin/env bash

set -u -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
command_file="${1:-$repo_root/commands_stencils.txt}"
output_dir="${2:-$repo_root/output}"
summary_file="$output_dir/summary.txt"
rm -rf $repo_root/.gt4py_cache

mkdir -p "$output_dir"
: > "$summary_file"

trim() {
  local value="$1"
  shopt -s extglob
  value="${value##+([[:space:]])}"
  value="${value%%+([[:space:]])}"
  shopt -u extglob
  printf '%s' "$value"
}

sanitize_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

strip_trailing_test_out_redirection() {
  printf '%s' "$1" | sed -E 's/[[:space:]]*>[[:space:]]*test_out\.txt[[:space:]]*$//'
}

run_count=0
passed_count=0
failed_count=0
skipped_count=0
label=""

{
  echo "Stencil command run summary"
  echo "Repository root: $repo_root"
  echo "Command file: $command_file"
  echo "Output directory: $output_dir"
  echo
} >> "$summary_file"

while IFS= read -r line || [[ -n "$line" ]]; do
  line="$(trim "$line")"
  [[ -z "$line" ]] && continue

  if [[ "$line" == \(* ]] || [[ "$line" == missing:* ]]; then
    continue
  fi

  if [[ "$line" == export\ * || "$line" == pytest\ * || "$line" == *pytest* ]]; then
    if [[ -z "$label" ]]; then
      ((skipped_count++))
      printf 'Skipped command without a preceding label: %s\n' "$line" >> "$summary_file"
      continue
    fi

    ((run_count++))
    safe_label="$(sanitize_name "$label")"
    [[ -z "$safe_label" ]] && safe_label="stencil"
    output_file="$output_dir/$(printf '%02d' "$run_count")_${safe_label}.txt"
    command_to_run="$(strip_trailing_test_out_redirection "$line")"

    {
      echo "=== $label ==="
      echo "Command: $command_to_run"
      echo "Started: $(date -Is)"
      echo
    } > "$output_file"

    start_seconds=$(date +%s)
    if bash -lc "$command_to_run" >> "$output_file" 2>&1; then
      status=0
      passed_count=$((passed_count + 1))
    else
      status=$?
      failed_count=$((failed_count + 1))
    fi
    end_seconds=$(date +%s)

    {
      echo
      echo "Exit code: $status"
      echo "Duration: $((end_seconds - start_seconds))s"
      echo "Output file: $output_file"
    } >> "$output_file"

    {
      echo "[$run_count] $label"
      echo "  status: $status"
      echo "  output: $(basename "$output_file")"
      echo "  duration: $((end_seconds - start_seconds))s"
      echo
    } >> "$summary_file"

    label=""
    continue
  fi

  label="$line"
done < "$command_file"

{
  echo "Totals"
  echo "  ran: $run_count"
  echo "  passed: $passed_count"
  echo "  failed: $failed_count"
  echo "  skipped: $skipped_count"
} >> "$summary_file"

printf 'Summary written to %s\n' "$summary_file"
printf 'Outputs written under %s\n' "$output_dir"

if [[ "$failed_count" -gt 0 ]]; then
  exit 1
fi