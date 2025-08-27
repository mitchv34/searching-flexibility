#!/bin/bash

# Live MPI Search Progress Monitor with Progress Bar
# Continuously tails the active job's intermediate JSON snapshots and renders a live progress bar.
# Usage:
#   ./live_progress.sh <JOB_ID>
#   ./live_progress.sh               # auto-detect most recent running MPI_Search job
# Options:
#   REFRESH_INTERVAL (env) seconds between scans (default 5)
#   WIDTH (env) progress bar width (default 50)
#   NO_COLOR=1 disable ANSI colors
#   WATCH=1 keep running after completion (otherwise exits when final file found)
#
# Dependencies: jq (for JSON parsing). Will fallback to grep/sed awk minimal parse if jq missing.

set -euo pipefail

REFRESH_INTERVAL=${REFRESH_INTERVAL:-5}
BAR_WIDTH=${WIDTH:-50}
NO_COLOR=${NO_COLOR:-0}
WATCH=${WATCH:-0}
BASE_DIR="/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/distributed_mpi_search"
RESULT_DIR="$BASE_DIR/output/results"
LOG_DIR="$BASE_DIR/output/logs"

if [[ ! -d "$RESULT_DIR" ]]; then
  mkdir -p "$RESULT_DIR"
fi

if [[ ! -d "$LOG_DIR" ]]; then
  mkdir -p "$LOG_DIR"
fi

# Colors
if [[ "$NO_COLOR" -eq 0 ]]; then
  C_RESET='\033[0m'
  C_BLUE='\033[34m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_RED='\033[31m'
  C_CYAN='\033[36m'
else
  C_RESET=''; C_BLUE=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN='';
fi

JOB_ID=${1:-}

# Auto-detect latest running job if not provided
if [[ -z "$JOB_ID" ]]; then
  JOB_ID=$(squeue -u "$USER" --name=MPI_Search --format="%A %T %V" 2>/dev/null | awk 'NR>1 {print $1}' | tail -n1 || true)
  if [[ -z "$JOB_ID" ]]; then
    echo "No running MPI_Search job detected." >&2
    exit 1
  fi
fi

echo -e "${C_CYAN}Live progress monitor for job ${JOB_ID}${C_RESET}" >&2

t_last_file=""
last_update_epoch=0

have_jq=1
if ! command -v jq >/dev/null 2>&1; then
  have_jq=0
  echo -e "${C_YELLOW}jq not found; using limited parser (install jq for richer output).${C_RESET}" >&2
fi

spinner=("|" "/" "-" "\\")
spin_i=0

# Render bar helper
render_bar(){
  local pct=$1
  local width=$2
  local filled=$(( (pct*width)/100 ))
  (( filled>width )) && filled=$width
  local empty=$(( width - filled ))
  printf '['
  printf '%0.s#' $(seq 1 $filled)
  printf '%0.s-' $(seq 1 $empty)
  printf ']'
}

# Minimal JSON parsing fallback (objective only)
extract_field_minimal(){
  local file=$1
  local key=$2
  grep -m1 "\"$key\"" "$file" | sed 's/.*: *//; s/[",]*$//' || echo "NA"
}

DEBUG_MONITOR=${DEBUG_MONITOR:-0}

while true; do
  spin_i=$(( (spin_i+1) % 4 ))
  # Pick latest job-specific snapshot
  # Prefer lightweight live status file if present for near-real-time updates
  live_file="$RESULT_DIR/mpi_search_results_job${JOB_ID}_live.json"
  if [[ -f "$live_file" ]]; then
    latest_file="$live_file"
  else
    latest_file=$(ls -1t "$RESULT_DIR"/mpi_search_results_job${JOB_ID}_*.json 2>/dev/null | head -n1 || true)
  fi
  state="PENDING"
  if squeue -j "$JOB_ID" --noheader >/dev/null 2>&1; then
    job_raw=$(squeue -j "$JOB_ID" --noheader 2>/dev/null | awk '{print $5}')
    [[ -n "$job_raw" ]] && state=$job_raw || state="COMPLETED"
  else
    state="COMPLETED"
  fi

  if [[ -z "$latest_file" ]]; then
    if [[ $DEBUG_MONITOR -eq 1 ]]; then
      # Emit a one–time diagnostic every loop while waiting
      match_count=$(ls -1 "$RESULT_DIR"/mpi_search_results_job${JOB_ID}_*.json 2>/dev/null | wc -l | awk '{print $1}')
      printf "\r${C_BLUE}Job:%s State:%s Waiting for first snapshot %s (debug: 0 files match pattern %s/mpi_search_results_job${JOB_ID}_*.json)${C_RESET}    " "$JOB_ID" "$state" "${spinner[$spin_i]}" "$RESULT_DIR"
    else
      printf "\r${C_BLUE}Job:%s State:%s Waiting for first snapshot %s${C_RESET}    " "$JOB_ID" "$state" "${spinner[$spin_i]}"
    fi
    sleep "$REFRESH_INTERVAL"
    continue
  fi

  if [[ "$latest_file" != "$t_last_file" ]]; then
    t_last_file="$latest_file"
    last_update_epoch=$(date +%s)
  fi

  if [[ $have_jq -eq 1 ]]; then
    # Prefer GA progress fields if present
    progress=$(jq -r '.ga_progress_pct // .progress_pct // (.progress*100) // 0' "$latest_file" 2>/dev/null | awk '{printf("%.2f", $1)}') || progress=0
    completed=$(jq -r '.completed_evaluations // 0' "$latest_file" 2>/dev/null || echo 0)
    total=$(jq -r '.total_evaluations // 0' "$latest_file" 2>/dev/null || echo 0)
    best_obj=$(jq -r '.best_objective // .bestObjective // "NA"' "$latest_file" 2>/dev/null)
    eta_h=$(jq -r '.eta_hours // 0' "$latest_file" 2>/dev/null | awk '{printf("%.2f", $1)}')
    # Fallback: if performance_metrics not present, use raw evaluation_rate (per second) * 60
    rate=$(jq -r '.performance_metrics.evaluations_per_minute // empty' "$latest_file" 2>/dev/null)
    if [[ -z "$rate" || "$rate" == "null" ]]; then
      rate_sec=$(jq -r '.evaluation_rate // 0' "$latest_file" 2>/dev/null)
      rate=$(awk -v rs="$rate_sec" 'BEGIN{printf("%.2f", rs*60)}')
    else
      rate=$(awk -v r="$rate" 'BEGIN{printf("%.2f", r)}')
    fi
    best_params=$(jq -r '.best_params | to_entries | map("\(.key)=\(.value)") | join(", ")' "$latest_file" 2>/dev/null | cut -c1-120)
    ga_gen=$(jq -r '.ga_generation // -1' "$latest_file" 2>/dev/null)
    ga_tot=$(jq -r '.ga_total_generations // -1' "$latest_file" 2>/dev/null)
    avg_eval=$(jq -r '.avg_time_per_eval // 0' "$latest_file" 2>/dev/null | awk '{printf("%.3f", $1)}')
    elapsed=$(jq -r '.elapsed_time_seconds // 0' "$latest_file" 2>/dev/null)
  else
    progress=0
    completed=$(extract_field_minimal "$latest_file" completed_evaluations)
    total=$(extract_field_minimal "$latest_file" total_evaluations)
    best_obj=$(extract_field_minimal "$latest_file" best_objective)
    eta_h="?"
    rate="?"
    best_params="(install jq)"
  ga_gen=-1
  ga_tot=-1
  avg_eval="?"
  elapsed=0
  fi

  if [[ "$total" == "0" || "$total" == "NA" ]]; then
    pct=0
  else
    pct=$(awk -v c="$completed" -v t="$total" 't>0 {printf "%.2f", (c/t)*100}' 2>/dev/null || echo 0)
    [[ -z "$pct" ]] && pct=0
  fi

  bar=$(render_bar ${pct%.*} $BAR_WIDTH)
  age=$(( $(date +%s) - last_update_epoch ))

  status_color=$C_YELLOW
  [[ $pct == 100.00 ]] && status_color=$C_GREEN
  [[ $state == PD ]] && status_color=$C_YELLOW
  [[ $state == R || $state == RUNNING ]] && status_color=$C_CYAN
  [[ $state == F || $state == FAILED ]] && status_color=$C_RED

  # GA generation progress string
  gen_str=""
  if [[ $ga_gen -ge 0 && $ga_tot -gt 0 ]]; then
    gen_pct=$(awk -v g="$ga_gen" -v gt="$ga_tot" 'gt>0 {printf "%.1f", (g/gt)*100}')
    gen_str=" Gen:${ga_gen}/${ga_tot}(${gen_pct}%)"
  fi

  # Derive per-generation stats if possible (approx)
  gen_rate="?"; gen_eta="?"; remaining_gens="?"
  if [[ $ga_gen -ge 0 && $ga_tot -gt 0 && $elapsed -gt 0 ]]; then
    # completed generations is ga_gen; assume population_size ~ (completed - maybe partial)
    remaining_gens=$(( ga_tot - ga_gen ))
    if [[ $completed -gt 0 ]]; then
      # Approx generation time = elapsed / max(ga_gen,1)
      gen_time=$(awk -v e=$elapsed -v g=$ga_gen 'g>0 {printf "%.2f", e/g}')
      gen_eta=$(awk -v gt=$gen_time -v rg=$remaining_gens 'rg>0 {printf "%.2f", (gt*rg)/3600}')
      gen_rate=$gen_time
    fi
  fi

  printf "\r%s Job:%s State:%s %s %5.1f%% %s%s | Eval:%s/%s | Best:%s | EvalRate:%s/m | AvgEval:%ss | ETA:%sh | GenTime:%ss RemGen:%s GenETA:%sh | Params:%s | %s" \
    "$status_color" "$JOB_ID" "$state" "${spinner[$spin_i]}" "$pct" "$bar" "$gen_str" "$completed" "$total" "$best_obj" "$rate" "$avg_eval" "$eta_h" "$gen_rate" "$remaining_gens" "$gen_eta" "$best_params" "$(basename "$latest_file")"
  printf "%s" "$C_RESET"

  # Exit criteria
  if [[ "$pct" == 100.00 || "$state" == COMPLETED ]]; then
    if [[ $WATCH -eq 0 ]]; then
      echo -e "\n${C_GREEN}Job complete or 100% progress. Exiting.${C_RESET}" >&2
      break
    fi
  fi

  sleep "$REFRESH_INTERVAL"
done
