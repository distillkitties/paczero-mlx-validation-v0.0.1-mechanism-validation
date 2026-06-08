#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
MODEL="${MODEL:-mlx-community/SmolLM-135M-4bit}"
SLUG="${SLUG:-smollm-135m-4bit}"
STEPS="${STEPS:-30}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-8}"
DEV_EXAMPLES="${DEV_EXAMPLES:-8}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-32}"
LAYERS="${LAYERS:-all}"
PROJECTIONS="${PROJECTIONS:-q_proj,v_proj}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16.0}"
MU="${MU:-0.05}"
LR="${LR:-0.05}"
CLIP="${CLIP:-25.0}"
EVAL_EVERY="${EVAL_EVERY:-5}"
NUM_SUBSETS="${NUM_SUBSETS:-126}"
PYTHON_BIN=""

hr() { printf '\n%s\n' "----------------------------------------------------------------------"; }
log() { printf '[paczero] %s\n' "$*"; }
warn() { printf '[paczero][warning] %s\n' "$*" >&2; }
fail() { printf '\n[paczero][error] %s\n' "$*" >&2; exit 1; }
run_cmd() { log "Running: $*"; "$@"; }

usage() {
  cat <<'USAGE'
PAC-Zero/ZPL MLX mechanism-validation artifact helper

Modes:
  quick                 Compile scripts, run negative control, rebuild aggregate report, print summary.
  aggregate             Rebuild aggregate report from included JSON files and print summary.
  negative-control      Run the ZPL negative-control audit only.
  install               Install/check MLX runtime packages for local reruns.
  local-sst2            Rerun the small SST-2 ZPL/MLX smoke task.
  local-squad           Rerun the small SQuAD ZPL/MLX smoke task.
  local-control-sst2    Rerun the small non-private SST-2 utility control.
  local-control-squad   Rerun the small non-private SQuAD utility control.
  local-all             Rerun all small local MLX smoke tasks, then aggregate.

Default local rerun settings:
  steps=30 train/dev/eval=8/8/32 M=126 rank=8 alpha=16 q_proj+v_proj layers=all

Claim boundary:
  This is a mechanism-validation artifact, not a paper-scale utility reproduction.
USAGE
}

ensure_repo_root() {
  [ -d scripts ] || fail "Run from the archive/repository root; ./scripts was not found."
  [ -f scripts/paczero_smollm_validation_aggregate.py ] || fail "Missing aggregate script."
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)";
  elif command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)";
  else PYTHON_BIN=""; fi
}

ensure_python() {
  find_python
  [ -n "$PYTHON_BIN" ] || fail "No Python found. Install Python 3.9+ or run this on macOS with /usr/bin/python3."
  log "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
  "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
}

install_deps() {
  ensure_python
  log "Installing/checking MLX runtime packages."
  run_cmd "$PYTHON_BIN" -m pip install --user --upgrade mlx mlx-lm huggingface_hub hf_transfer safetensors numpy datasets
  "$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ['mlx', 'mlx_lm', 'huggingface_hub', 'safetensors', 'numpy', 'datasets'] if importlib.util.find_spec(name) is None]
if missing:
    print('missing:', ', '.join(missing))
    raise SystemExit(1)
print('MLX/runtime imports: ok')
PY
}

compile_scripts() {
  ensure_python
  run_cmd "$PYTHON_BIN" -m py_compile scripts/*.py
}

run_negative_control() {
  ensure_python
  run_cmd "$PYTHON_BIN" scripts/paczero_zpl_negative_control.py
}

print_summary() {
  ensure_python
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path('benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_aggregate_results.json')
report = Path('benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_report.md')
neg_path = Path('benchmark-results/paczero-smollm-validation-aggregate/zpl_negative_control_results.json')
print('\n' + '=' * 72)
print('PAC-ZERO/ZPL MLX v0.0.1 MECHANISM-VALIDATION SUMMARY')
print('=' * 72)
if not path.exists():
    print(f'Missing aggregate JSON: {path}')
    raise SystemExit(0)
data = json.loads(path.read_text())
def yn(x): return 'PASS' if bool(x) else 'FAIL'
def fmt(x): return 'n/a' if x is None else (f'{x:.6g}' if isinstance(x, float) else str(x))
print(f'Overall success: {yn(data.get("success"))}')
print('Claim boundary: mechanism validation only; not paper-scale utility reproduction.')
print(f'Original aggregate claim: {data.get("claim", "n/a")}')
print()
checks = data.get('aggregate_checks') or {}
if checks:
    print('Aggregate mechanism/audit checks:')
    for key in sorted(checks):
        print(f'  - {key}: {yn(checks[key])}')
    print()
neg = data.get('negative_control') or {}
if neg:
    print('Negative control:')
    print(f'  - success: {yn(neg.get("success"))}')
    print(f'  - good ZPL passes audit: {yn((neg.get("checks") or {}).get("good_zpl_release_passes_audit"))}')
    print(f'  - bad S*-dependent release fails audit: {yn((neg.get("checks") or {}).get("bad_secret_dependent_release_fails_audit"))}')
    print(f'  - conclusion: {neg.get("conclusion", "n/a")}')
    print()
print('Task smoke checks:')
for task in data.get('tasks') or []:
    print(f'  - {task.get("task", "unknown")}: {yn(task.get("success"))}; M={fmt(task.get("num_subsets_M"))}; membership={fmt(task.get("membership_counts"))}; privacy_audit={yn(task.get("privacy_transcript_audit_passed"))}; violations={fmt(task.get("release_rule_violation_count"))}')
print()
print('Limitations / non-claims:')
for item in data.get('non_claims_limitations') or []:
    print(f'  - {item}')
print()
print('Output files:')
for p in [path, report, neg_path]:
    print(f'  - {p} ({p.stat().st_size} bytes)' if p.exists() else f'  - {p} (missing)')
print('=' * 72)
PY
}

run_aggregate() {
  ensure_python
  run_cmd "$PYTHON_BIN" scripts/paczero_smollm_validation_aggregate.py
  print_summary
}

run_validation_task() {
  install_deps
  task="$1"; seed="$2"
  out_dir="benchmark-results/paczero-smollm-validation/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-validation-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  run_cmd "$PYTHON_BIN" scripts/paczero_mlxlm_faithful_adaptation.py \
    --model "$MODEL" --slug "$SLUG" --task "$task" --projections "$PROJECTIONS" --layers "$LAYERS" \
    --rank "$RANK" --alpha "$ALPHA" --seed "$seed" --steps "$STEPS" \
    --train-examples "$TRAIN_EXAMPLES" --dev-examples "$DEV_EXAMPLES" --eval-examples "$EVAL_EXAMPLES" \
    --num-subsets "$NUM_SUBSETS" --mu "$MU" --lr "$LR" --clip "$CLIP" --eval-every "$EVAL_EVERY" \
    --json-out "$out_dir/smollm_validation_results.json" \
    --adapter-out "$adapter_dir/all_layers_qv_lora_rank8_alpha16.npz"
}

run_utility_control_task() {
  install_deps
  task="$1"; seed="$2"
  out_dir="benchmark-results/paczero-smollm-utility-control/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-utility-control-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  run_cmd "$PYTHON_BIN" scripts/paczero_smollm_nonprivate_utility_control.py \
    --model "$MODEL" --slug "$SLUG" --task "$task" --projections "$PROJECTIONS" --layers "$LAYERS" \
    --rank "$RANK" --alpha "$ALPHA" --seed "$seed" --steps "$STEPS" \
    --train-examples "$TRAIN_EXAMPLES" --dev-examples "$DEV_EXAMPLES" --eval-examples "$EVAL_EXAMPLES" \
    --mu "$MU" --lr "$LR" --eval-every "$EVAL_EVERY" \
    --json-out "$out_dir/nonprivate_utility_control_results.json" \
    --adapter-out "$adapter_dir/nonprivate_all_layers_qv_lora_rank8_alpha16.npz"
}

ensure_repo_root
case "$MODE" in
  help|-h|--help) usage ;;
  quick) compile_scripts; run_negative_control; run_aggregate; log "Quick check complete." ;;
  aggregate) run_aggregate ;;
  negative-control) run_negative_control ;;
  install) install_deps ;;
  local-sst2) run_validation_task sst2 20260615 ;;
  local-squad) run_validation_task squad 20260616 ;;
  local-control-sst2) run_utility_control_task sst2 20260618 ;;
  local-control-squad) run_utility_control_task squad 20260619 ;;
  local-all) run_validation_task sst2 20260615; run_validation_task squad 20260616; run_utility_control_task sst2 20260618; run_utility_control_task squad 20260619; run_negative_control; run_aggregate; log "Full local smoke run complete." ;;
  *) echo "Unknown mode: $MODE" >&2; usage >&2; exit 2 ;;
esac
