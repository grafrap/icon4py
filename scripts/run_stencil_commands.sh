#!/usr/bin/env bash

set -u -o pipefail

usage() {
  cat <<'EOF'
Usage: run_stencil_commands.sh [options] [command_file] [output_dir]

Options:
  -c, --command-file PATH   Path to the stencil command list
  -o, --output-dir PATH     Directory for per-stencil outputs and summary
  -j, --jobs N              Number of stencil commands to run in parallel
  -h, --help                Show this help message
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
command_file="$repo_root/commands_stencils.txt"
output_dir="$repo_root/output"
jobs="${JOBS:-1}"

positionals=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--command-file)
      command_file="${2:?Missing value for --command-file}"
      shift 2
      ;;
    -o|--output-dir)
      output_dir="${2:?Missing value for --output-dir}"
      shift 2
      ;;
    -j|--jobs)
      jobs="${2:?Missing value for --jobs}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        positionals+=("$1")
        shift
      done
      break
      ;;
    *)
      positionals+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${positionals[0]:-}" ]]; then
  command_file="${positionals[0]}"
fi

if [[ -n "${positionals[1]:-}" ]]; then
  output_dir="${positionals[1]}"
fi

if ! [[ "$jobs" =~ ^[0-9]+$ ]] || [[ "$jobs" -lt 1 ]]; then
  jobs=1
fi

summary_file="$output_dir/summary.txt"

mkdir -p "$output_dir"
: > "$summary_file"

meta_dir="$(mktemp -d "$output_dir/.stencil-run.XXXXXX")"

cleanup() {
  rm -rf "$meta_dir"
}
trap cleanup EXIT

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
declare -a task_labels=()
declare -a task_commands=()
declare -a task_outputs=()
declare -a task_status_files=()
declare -a task_duration_files=()

{
  echo "Stencil command run summary"
  echo "Repository root: $repo_root"
  echo "Command file: $command_file"
  echo "Output directory: $output_dir"
  echo "Parallel jobs: $jobs"
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

    task_labels+=("$label")
    task_commands+=("$command_to_run")
    task_outputs+=("$output_file")
    task_status_files+=("$meta_dir/$(printf '%02d' "$run_count").status")
    task_duration_files+=("$meta_dir/$(printf '%02d' "$run_count").duration")

    label=""
    continue
  fi

  label="$line"
done < "$command_file"

rm -rf "$repo_root/.gt4py_cache"

run_task() {
  local task_label="$1"
  local command_to_run="$2"
  local output_file="$3"
  local status_file="$4"
  local duration_file="$5"

  {
    echo "=== $task_label ==="
    echo "Command: $command_to_run"
    echo "Started: $(date -Is)"
    echo
  } > "$output_file"

  local start_seconds end_seconds status
  start_seconds=$(date +%s)
  if bash -lc "$command_to_run" >> "$output_file" 2>&1; then
    status=0
  else
    status=$?
  fi
  end_seconds=$(date +%s)

  printf '%s\n' "$status" > "$status_file"
  printf '%s\n' "$((end_seconds - start_seconds))" > "$duration_file"

  {
    echo
    echo "Exit code: $status"
    echo "Duration: $((end_seconds - start_seconds))s"
    echo "Output file: $output_file"
  } >> "$output_file"
}

for idx in "${!task_labels[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$jobs" ]]; do
    wait -n || true
  done

  run_task "${task_labels[$idx]}" "${task_commands[$idx]}" "${task_outputs[$idx]}" "${task_status_files[$idx]}" "${task_duration_files[$idx]}" &
done

wait

for idx in "${!task_labels[@]}"; do
  status="$(cat "${task_status_files[$idx]}")"
  duration="$(cat "${task_duration_files[$idx]}")"

  if [[ "$status" -eq 0 ]]; then
    passed_count=$((passed_count + 1))
  else
    failed_count=$((failed_count + 1))
  fi

  {
    echo "[$((idx + 1))] ${task_labels[$idx]}"
    echo "  status: $status"
    echo "  output: $(basename "${task_outputs[$idx]}")"
    echo "  duration: ${duration}s"
    echo
  } >> "$summary_file"
done

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