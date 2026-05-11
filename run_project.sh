#!/usr/bin/env bash
# ============================================================================
# run_project.sh
#
# One-shot driver for the four-way comparison experiment. Loads the same
# cluster toolchain that setup_env.sh used, activates the .venv (Python 3.12
# + cu130 torch stack + vLLM 0.18.0 from source), serves a single local vLLM
# instance, and runs evaluations:
#
#   4 controllers (vanilla tool-calling, Act, ReAct, PRAXIS)
#   x 3 benchmarks (tau-retail, tau-airline, BFCL V4)
#
# All four controllers share the same in-process loop, model, temperature,
# tool schemas, max steps, and truncation budget — so the only varying axis
# is the controller. PRAXIS keeps native function-calling but adds audited
# procedural experience retrieval (HybridRetriever + TacticalPlaybook) and
# write-only SABER-style mutation reflection.
# Produces outputs/summary/{summary.json,summary.md}.
#
# The venv created by setup_env.sh must already exist.
#
# Usage examples
# --------------
#   bash run_project.sh                       # auto-detect GPUs and use sensible defaults
#   bash run_project.sh --gpus 0              # force 1-GPU mode
#   bash run_project.sh --gpus 0,1            # force 2-GPU tensor-parallel mode
#   bash run_project.sh --model 7b            # explicit 7B (for 40 GB GPUs or max speed)
#   bash run_project.sh --model 14b           # 14B (default on 1×80 GB A100/H100)
#   bash run_project.sh --model 32b           # 32B (recommended on 2× GPUs)
#   bash run_project.sh --profile smoke       # tiny sanity sweep (~10–15 min)
#   bash run_project.sh --profile small       # 15 tasks × 3 trials (~1–2 h)
#   bash run_project.sh --profile medium      # 30 tasks × 3 trials (~3–6 h)  [default]
#   bash run_project.sh --profile full        # 50 tasks × 4 trials (~8–14 h, full 15h budget)
#   bash run_project.sh --tau-tasks 20 --tau-trials 2 --bfcl-limit 10
#   bash run_project.sh --controllers baseline,react,praxis --skip-bfcl
#   bash run_project.sh --tau-only retail --controllers praxis
#   bash run_project.sh --dry-run             # print resolved config and exit
#
# Run `bash run_project.sh --help` for the full flag reference. Every CLI
# flag also has an environment-variable equivalent (see --help) so the
# script is friendly to job schedulers that prefer env vars over argv.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

log() { echo "[run] $*"; }

# ---------------------------------------------------------------------------
# CLI argument parsing
#
# All flags also accept an environment-variable form. Precedence is:
#     CLI flag > environment variable > built-in default.
# ---------------------------------------------------------------------------
print_help() {
  cat <<'HELP'
Usage: bash run_project.sh [OPTIONS]

GPU and model:
  --gpus LIST            Comma-separated GPU indices (default: auto-detect via nvidia-smi).
                         Examples: "0", "0,1". Tensor-parallel size is derived from the count.
                         Env: GPUS=0,1
  --model TIER           Model tier: 7b | 14b | 32b | auto (default: auto).
                         auto = 32b when ≥2 GPUs are available, else 14b.
                         7b   = Qwen2.5-7B-Instruct → Qwen3-4B fallbacks.
                                (use on 40 GB GPUs or if you need maximum speed)
                         14b  = Qwen2.5-14B-Instruct → 7B → 4B fallbacks.
                                (default for 1×80 GB A100/H100; better JSON quality)
                         32b  = Qwen2.5-32B-Instruct → 7B → 4B fallbacks.
                                (recommended on ≥2 GPUs with TP=2)
                         Env: MODEL_TIER=14b
  --tp N                 Override tensor-parallel-size (default: number of --gpus).
                         Env: TENSOR_PARALLEL_SIZE=2
  --max-model-len N      Max context length in tokens (default: 32768 for all tiers).
                         All tiers default to 32K; override only if you hit OOM.
                         Env: MAX_MODEL_LEN=32768
  --gpu-mem-util F       GPU memory utilization 0..1 (default 0.80, lowered for 32B/TP=1).
                         Env: GPU_MEM_UTIL=0.85

Workload:
  --profile NAME         Workload profile: smoke | small | medium | full (default: medium).
                           smoke   =   5 tasks ×  1 trial,  bfcl=10,  conc=2  (~10–15 min)
                           small   =  15 tasks ×  3 trials, bfcl=20,  conc=4  (~1–2 h)
                           medium  =  30 tasks ×  3 trials, bfcl=40,  conc=4  (~3–6 h)
                           full    =  50 tasks ×  4 trials, bfcl=80,  conc=8  (~8–14 h)
                         Env: PROFILE=medium
  --tau-tasks N          Override τ-bench tasks per env (start=0, end=N).
                         Env: TAU_END_INDEX=N (uses TAU_START_INDEX=0)
  --tau-trials N         Override τ-bench trial count.       Env: TAU_NUM_TRIALS=N
  --bfcl-limit N         Override BFCL task count total.     Env: BFCL_LIMIT=N
  --bfcl-data-dir DIR    Path to local BFCL data directory.  Env: BFCL_DATA_DIR=DIR
  --bfcl-categories CAT  Comma-separated BFCL categories.    Env: BFCL_CATEGORIES=...
  --max-concurrency N    Override per-runner max concurrency. Env: MAX_CONCURRENCY=N
  --tau-max-steps N      Cap τ-bench steps per trajectory.    Env: TAU_MAX_STEPS=N
  --bfcl-max-steps N     Cap BFCL steps per trajectory.       Env: BFCL_MAX_STEPS=N

Controller filter:
  --controllers LIST     Comma-separated subset of {baseline,act,react,praxis}.
                         Default: baseline,act,react,praxis. Env: CONTROLLERS=...
  --skip-bfcl            Skip BFCL cells; run only τ-bench retail+airline.
                         Env: SKIP_BFCL=1
  --praxis-prewarm       Run PRAXIS on the retail TRAIN split first to seed the
                         experience bank, then run all agents on the TEST split.
                         Use this for warm-start results. Env: PRAXIS_PREWARM=1
  --rex-prewarm          Alias for --praxis-prewarm (backward compatible).
  --tau-only ENV         Run only one τ-bench env (retail|airline). Env: TAU_ONLY=...

Server:
  --port N               vLLM port (default: 8001). Env: PORT=N
  --served-name STR      vLLM served-model-name (default: qwen-agent). Env: SERVED_NAME=...

GPU-specific venvs:
  By default, run_project.sh auto-detects the GPU type (A100, H100, H200) and
  uses a pre-built venv (.venv-a100, .venv-h100, .venv-h200). This avoids
  rebuilding vLLM when switching between GPU types.

  To pre-build a GPU-specific venv, run:
    VENV_DIR=.venv-a100 bash setup_env.sh   # for A100
    VENV_DIR=.venv-h100 bash setup_env.sh   # for H100
    VENV_DIR=.venv-h200 bash setup_env.sh   # for H200

  To override and use a specific venv for run_project.sh:
    VENV_DIR=/path/to/venv bash run_project.sh

Misc:
  --dry-run              Print resolved configuration and exit. Env: DRY_RUN=1
  --help, -h             Show this help and exit.
HELP
}

# Default flag values (overridable via env or CLI).
GPUS_OPT="${GPUS:-}"
MODEL_TIER="${MODEL_TIER:-auto}"
TP_OPT="${TENSOR_PARALLEL_SIZE:-}"
PROFILE="${PROFILE:-medium}"
CONTROLLERS_OPT="${CONTROLLERS:-baseline,act,react,praxis}"
SKIP_BFCL="${SKIP_BFCL:-0}"
TAU_ONLY="${TAU_ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"
# PRAXIS_PREWARM=1: run praxis on the train split first (retail only) to seed the
# experience bank before the main evaluation pass on the test split.  Disabled
# by default so a first run still produces valid cold-start numbers.
PRAXIS_PREWARM="${PRAXIS_PREWARM:-${REX_PREWARM:-0}}"
REX_PREWARM="$PRAXIS_PREWARM"  # keep backward-compat alias

# ---------------------------------------------------------------------------
# Three-phase architecture (Phase A → Phase B → Phase C).
#
#   PHASE A — Baseline evaluation (CLEAN, no recording)
#     Run every controller in $EVAL_CONTROLLERS on the held-out TEST split
#     with --promote-runtime-memory=never. Their metrics are reported as-is
#     and their trajectories are NEVER distilled.
#
#   PHASE B — Experience harvest (donor cohort, TRAIN only)
#     Run every controller in $DONOR_CONTROLLERS on the TRAIN split (or the
#     train slice of the internal split) with --promote-runtime-memory=auto.
#     Each donor contributes Process Memory Cards from a different prompting
#     prior so PRAXIS reasons from a *diverse* experience pool rather than
#     learning only from itself or one ReAct baseline.
#
#   PHASE C — PRAXIS evaluation
#     With the warm, diverse bank from Phase B, PRAXIS runs on the TEST
#     split. Promotion stays gated on test, so it cannot leak.
#
# Defaults:
#   EVAL_CONTROLLERS  = baseline,act,react              (paper baselines)
#   DONOR_CONTROLLERS = act,react,cot,plan-solve,reflexion-lite,praxis
#                       (diverse prompting priors; praxis self-donates on
#                        train so its bank also captures its own successes)
#
# Override via env vars. Setting RUN_PHASE_A=0 / RUN_PHASE_B=0 / RUN_PHASE_C=0
# skips that phase (useful for re-running just one piece).
EVAL_CONTROLLERS="${EVAL_CONTROLLERS:-baseline,act,react}"
DONOR_CONTROLLERS="${DONOR_CONTROLLERS:-act,react,cot,plan-solve,reflexion-lite,praxis}"
RUN_PHASE_A="${RUN_PHASE_A:-1}"
RUN_PHASE_B="${RUN_PHASE_B:-1}"
RUN_PHASE_C="${RUN_PHASE_C:-1}"
# PRAXIS_RESET_BANK=1 wipes REX_RUNTIME_DIR before Phase B so the donor
# harvest writes into a clean bank (recommended for paper-grade runs).
PRAXIS_RESET_BANK="${PRAXIS_RESET_BANK:-0}"

# Per-knob overrides (these take precedence over the profile).
TAU_END_INDEX_OPT=""
TAU_NUM_TRIALS_OPT=""
BFCL_LIMIT_OPT=""
BFCL_DATA_DIR_OPT=""
BFCL_CATEGORIES_OPT=""
MAX_CONCURRENCY_OPT=""
TAU_MAX_STEPS_OPT=""
BFCL_MAX_STEPS_OPT=""
MAX_MODEL_LEN_OPT=""
GPU_MEM_UTIL_OPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)            GPUS_OPT="$2"; shift 2 ;;
    --gpus=*)          GPUS_OPT="${1#*=}"; shift ;;
    --model)           MODEL_TIER="$2"; shift 2 ;;
    --model=*)         MODEL_TIER="${1#*=}"; shift ;;
    --tp)              TP_OPT="$2"; shift 2 ;;
    --tp=*)            TP_OPT="${1#*=}"; shift ;;
    --profile)         PROFILE="$2"; shift 2 ;;
    --profile=*)       PROFILE="${1#*=}"; shift ;;
    --tau-tasks)       TAU_END_INDEX_OPT="$2"; shift 2 ;;
    --tau-tasks=*)     TAU_END_INDEX_OPT="${1#*=}"; shift ;;
    --tau-trials)      TAU_NUM_TRIALS_OPT="$2"; shift 2 ;;
    --tau-trials=*)    TAU_NUM_TRIALS_OPT="${1#*=}"; shift ;;
    --bfcl-limit)      BFCL_LIMIT_OPT="$2"; shift 2 ;;
    --bfcl-limit=*)    BFCL_LIMIT_OPT="${1#*=}"; shift ;;
    --bfcl-data-dir)   BFCL_DATA_DIR_OPT="$2"; shift 2 ;;
    --bfcl-data-dir=*) BFCL_DATA_DIR_OPT="${1#*=}"; shift ;;
    --bfcl-categories) BFCL_CATEGORIES_OPT="$2"; shift 2 ;;
    --bfcl-categories=*) BFCL_CATEGORIES_OPT="${1#*=}"; shift ;;
    --max-concurrency) MAX_CONCURRENCY_OPT="$2"; shift 2 ;;
    --max-concurrency=*) MAX_CONCURRENCY_OPT="${1#*=}"; shift ;;
    --tau-max-steps)   TAU_MAX_STEPS_OPT="$2"; shift 2 ;;
    --tau-max-steps=*) TAU_MAX_STEPS_OPT="${1#*=}"; shift ;;
    --bfcl-max-steps)  BFCL_MAX_STEPS_OPT="$2"; shift 2 ;;
    --bfcl-max-steps=*) BFCL_MAX_STEPS_OPT="${1#*=}"; shift ;;
    --max-model-len)   MAX_MODEL_LEN_OPT="$2"; shift 2 ;;
    --max-model-len=*) MAX_MODEL_LEN_OPT="${1#*=}"; shift ;;
    --gpu-mem-util)    GPU_MEM_UTIL_OPT="$2"; shift 2 ;;
    --gpu-mem-util=*)  GPU_MEM_UTIL_OPT="${1#*=}"; shift ;;
    --controllers)     CONTROLLERS_OPT="$2"; shift 2 ;;
    --controllers=*)   CONTROLLERS_OPT="${1#*=}"; shift ;;
    --skip-bfcl)       SKIP_BFCL=1; shift ;;
    --praxis-prewarm)  PRAXIS_PREWARM=1; REX_PREWARM=1; shift ;;
    --rex-prewarm)     PRAXIS_PREWARM=1; REX_PREWARM=1; shift ;;  # backward compat
    --tau-only)        TAU_ONLY="$2"; shift 2 ;;
    --tau-only=*)      TAU_ONLY="${1#*=}"; shift ;;
    --port)            PORT_OPT="$2"; shift 2 ;;
    --port=*)          PORT_OPT="${1#*=}"; shift ;;
    --served-name)     SERVED_NAME_OPT="$2"; shift 2 ;;
    --served-name=*)   SERVED_NAME_OPT="${1#*=}"; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --help|-h)         print_help; exit 0 ;;
    --)                shift; break ;;
    *) echo "Unknown flag: $1" >&2; echo "Use --help for usage." >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
detect_gpu_csv() {
  if [[ -n "$GPUS_OPT" ]]; then
    echo "$GPUS_OPT"; return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n >= 2 )); then echo "0,1"
    elif [[ "$n" =~ ^[0-9]+$ ]] && (( n == 1 )); then echo "0"
    else echo "0"  # last-resort default; vLLM will error if no GPU present
    fi
  else
    echo "0"
  fi
}
GPUS_CSV="$(detect_gpu_csv)"
NUM_GPUS=$(awk -F, '{print NF}' <<<"$GPUS_CSV")
if [[ -n "$TP_OPT" ]]; then
  TP="$TP_OPT"
else
  TP="$NUM_GPUS"
fi

# ---------------------------------------------------------------------------
# Tier resolution: auto → 32b (≥2 GPUs) | 14b (1 GPU, 80 GB)
#
# Why 14B over 7B on a single 80 GB A100/H100?
#   • 14B BF16 weights ≈ 28 GB + KV cache at max_model_len=32768 ≈ 24 GB → ~52 GB total
#     (concurrency=4 × 32K tokens, paged pool). Fits in 80 GB at 0.85 mem util.
#   • PRAXIS relies on native function-calling plus short reflection. 14B has
#     measurably better tool argument reliability than 7B while still fitting
#     a one-GPU evaluation budget.
#   • Speed cost: ~1.8–2.0× slower than 7B per call, still fits medium profile
#     in ~5–9 h on a single 80 GB GPU.
#
# Why 32K context for all tiers?
#   • τ-bench trajectories can be long: 30 tasks × up to 30 steps, each step
#     appending tool-call JSON + observation. At 12K tokens, late-trajectory
#     turns get truncated and the agent loses earlier DB-confirmed facts.
#   • Memory budget per tier at 32K:
#       7B  on 40 GB (0.80): 14 GB weights + ~18 GB KV pool → ~32 GB < 32 GB (✓)
#       14B on 80 GB (0.85): 28 GB weights + ~24 GB KV pool → ~52 GB < 68 GB (✓)
#       32B on 2×80 GB TP=2 (0.85): 32 GB/GPU weights + ~36 GB/GPU KV → ~68 GB/GPU < 68 GB (✓)
#   • Override with --max-model-len if you hit OOM on unusual hardware.
#
# Use --model 7b explicitly if you have a 40 GB GPU or need maximum speed.
# ---------------------------------------------------------------------------
case "$MODEL_TIER" in
  auto)
    if (( NUM_GPUS >= 2 )); then MODEL_TIER="32b"; else MODEL_TIER="14b"; fi
    ;;
  7b|14b|32b) ;;
  *) echo "ERROR: --model must be one of: 7b, 14b, 32b, auto (got '$MODEL_TIER')" >&2; exit 1 ;;
esac

# Pick model candidate chain by tier. The first that serves successfully
# is used for ALL controllers.
if [[ "$MODEL_TIER" == "32b" ]]; then
  MODEL_CANDIDATES=(
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507-FP8"
  )
  # 32B TP=2 on 2×80 GB: 32K context; KV pool ≈ 36 GB/GPU — fits within 68 GB/GPU.
  # TP=1 (fallback, single GPU, not recommended): uses higher mem util to compensate.
  TIER_DEFAULT_MAX_LEN=32768
  if (( TP <= 1 )); then TIER_DEFAULT_MEM_UTIL="0.92"; else TIER_DEFAULT_MEM_UTIL="0.85"; fi
elif [[ "$MODEL_TIER" == "14b" ]]; then
  MODEL_CANDIDATES=(
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507-FP8"
  )
  # 14B on 80 GB: 32K context, KV pool ≈ 24 GB (conc=4) → ~52 GB total < 68 GB.
  TIER_DEFAULT_MAX_LEN=32768
  TIER_DEFAULT_MEM_UTIL="0.85"
else
  # 7b — explicit preference for speed or 40 GB GPUs.
  MODEL_CANDIDATES=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507-FP8"
    "Qwen/Qwen3-4B-Instruct-2507"
  )
  # 7B on 40 GB: 32K context, KV pool ≈ 18 GB (conc=4) → ~32 GB total < 32 GB (tight; 0.80).
  TIER_DEFAULT_MAX_LEN=32768
  TIER_DEFAULT_MEM_UTIL="0.80"
fi

# ---------------------------------------------------------------------------
# Profile resolution (workload sizing)
# ---------------------------------------------------------------------------
case "$PROFILE" in
  smoke)   P_TAU_END=5;  P_TAU_TRIALS=1; P_BFCL_LIMIT=10; P_CONC=2; P_TAU_STEPS=20; P_BFCL_STEPS=15 ;;
  small)   P_TAU_END=15; P_TAU_TRIALS=3; P_BFCL_LIMIT=20; P_CONC=4; P_TAU_STEPS=30; P_BFCL_STEPS=20 ;;
  medium)  P_TAU_END=30; P_TAU_TRIALS=3; P_BFCL_LIMIT=40; P_CONC=4; P_TAU_STEPS=30; P_BFCL_STEPS=20 ;;
  full)    P_TAU_END=50; P_TAU_TRIALS=4; P_BFCL_LIMIT=80; P_CONC=8; P_TAU_STEPS=30; P_BFCL_STEPS=20 ;;
  *) echo "ERROR: --profile must be smoke|small|medium|full (got '$PROFILE')" >&2; exit 1 ;;
esac

# Multi-GPU bumps the default concurrency since vLLM batches across replicas.
if (( NUM_GPUS >= 2 )); then
  P_CONC=$(( P_CONC * 2 ))
fi

# Apply per-knob overrides (CLI > env > profile).
TAU_END_INDEX="${TAU_END_INDEX_OPT:-${TAU_END_INDEX:-$P_TAU_END}}"
TAU_NUM_TRIALS="${TAU_NUM_TRIALS_OPT:-${TAU_NUM_TRIALS:-$P_TAU_TRIALS}}"
BFCL_LIMIT="${BFCL_LIMIT_OPT:-${BFCL_LIMIT:-$P_BFCL_LIMIT}}"
TAU_MAX_CONCURRENCY="${MAX_CONCURRENCY_OPT:-${MAX_CONCURRENCY:-${TAU_MAX_CONCURRENCY:-$P_CONC}}}"
BFCL_MAX_CONCURRENCY="${MAX_CONCURRENCY_OPT:-${MAX_CONCURRENCY:-${BFCL_MAX_CONCURRENCY:-$P_CONC}}}"
TAU_MAX_STEPS="${TAU_MAX_STEPS_OPT:-${TAU_MAX_STEPS:-$P_TAU_STEPS}}"
BFCL_MAX_STEPS="${BFCL_MAX_STEPS_OPT:-${BFCL_MAX_STEPS:-$P_BFCL_STEPS}}"

# Tier-aware vLLM serving knobs.
PRIMARY_MAX_LEN="${MAX_MODEL_LEN_OPT:-${MAX_MODEL_LEN:-$TIER_DEFAULT_MAX_LEN}}"
PRIMARY_MEM_UTIL="${GPU_MEM_UTIL_OPT:-${GPU_MEM_UTIL:-$TIER_DEFAULT_MEM_UTIL}}"

# ---------------------------------------------------------------------------
# Validate --controllers / --tau-only up-front so typos fail fast (also
# during --dry-run).
# ---------------------------------------------------------------------------
ALLOWED_CONTROLLERS=("baseline" "act" "react" "praxis" "rex")
IFS=',' read -r -a REQUESTED_CONTROLLERS <<<"$CONTROLLERS_OPT"
for c in "${REQUESTED_CONTROLLERS[@]}"; do
  c="$(echo "$c" | tr -d '[:space:]')"
  [[ -z "$c" ]] && continue
  ok=0
  for a in "${ALLOWED_CONTROLLERS[@]}"; do [[ "$a" == "$c" ]] && { ok=1; break; }; done
  if [[ $ok -eq 0 ]]; then
    echo "ERROR: unknown controller '$c'. Allowed: ${ALLOWED_CONTROLLERS[*]}" >&2
    exit 1
  fi
done
case "${TAU_ONLY:-}" in
  ""|"both"|retail|airline) ;;
  *) echo "ERROR: --tau-only must be retail|airline (got '$TAU_ONLY')" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Early dry-run gate.
#
# Print the GPU/tier/profile config and exit before touching the venv or
# loading cluster modules. Useful on a login node to validate the flag
# combinations before submitting the actual job.
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
  cat <<INFO
---------- Resolved configuration (dry-run) ----------
GPUS=$GPUS_CSV   NUM_GPUS=$NUM_GPUS   TP=$TP
MODEL_TIER=$MODEL_TIER
  primary_max_len=$PRIMARY_MAX_LEN   gpu_mem_util=$PRIMARY_MEM_UTIL
PROFILE=$PROFILE
  TAU:  tasks=[${TAU_START_INDEX:-0}..${TAU_END_INDEX}) trials=$TAU_NUM_TRIALS conc=$TAU_MAX_CONCURRENCY max_steps=$TAU_MAX_STEPS
  BFCL: limit=$BFCL_LIMIT conc=$BFCL_MAX_CONCURRENCY max_steps=$BFCL_MAX_STEPS
CONTROLLERS=$CONTROLLERS_OPT
SKIP_BFCL=$SKIP_BFCL
TAU_ONLY=${TAU_ONLY:-(both)}
MODEL_CANDIDATES:
$(printf '  - %s\n' "${MODEL_CANDIDATES[@]}")
------------------------------------------------------
Dry-run requested; exiting before cluster setup.
INFO
  exit 0
fi

# ---------------------------------------------------------------------------
# Paths (must match setup_env.sh)
# ---------------------------------------------------------------------------
: "${PROJECT_SCRATCH:=${REPO_ROOT}/.scratch}"
export PROJECT_SCRATCH

# If VENV_DIR is explicitly set, use it as-is.
# Otherwise, we will select a GPU-specific venv after GPU detection.
if [[ -z "${VENV_DIR:-}" ]]; then
  VENV_DIR="${REPO_ROOT}/.venv"
fi

EXTERNAL_DIR="${EXTERNAL_DIR:-${REPO_ROOT}/external}"
TAU_DIR="${TAU_DIR:-${EXTERNAL_DIR}/tau-bench}"
BFCL_DATA_DIR="${BFCL_DATA_DIR_OPT:-${BFCL_DATA_DIR:-${EXTERNAL_DIR}/bfcl}}"

export HF_HOME="${HF_HOME:-${PROJECT_SCRATCH}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_SCRATCH}/torch_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_SCRATCH}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torch/inductor}"
export TMPDIR="${TMPDIR:-${PROJECT_SCRATCH}/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PROJECT_SCRATCH}/torch_extensions}"

mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# HF token (gated model downloads). Override by exporting HF_TOKEN.
# ---------------------------------------------------------------------------
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN"

# ---------------------------------------------------------------------------
# Cluster modules (same pinning as setup_env.sh)
# ---------------------------------------------------------------------------
GCC_MODULE="${GCC_MODULE:-gcc-13.2.0-gcc-12.1.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda-13.0.1-gcc-13.2.0}"

ensure_module_cmd () {
  if command -v module >/dev/null 2>&1; then return 0; fi
  for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/lmod.sh; do
    if [[ -f "$init" ]]; then
      # shellcheck disable=SC1090
      source "$init"; break
    fi
  done
  command -v module >/dev/null 2>&1
}

if ensure_module_cmd; then
  log "Loading cluster modules: $GCC_MODULE + $CUDA_MODULE"
  module purge || true
  module load "$GCC_MODULE" || log "WARNING: module load $GCC_MODULE failed"
  module load "$CUDA_MODULE" || log "WARNING: module load $CUDA_MODULE failed"
fi

unset PYTHONPATH
unset TRANSFORMERS_CACHE
unset VLLM_CACHE_DIR
# Don't leak cluster NCCL/CUDA libs onto the cu130 wheels.
unset LD_LIBRARY_PATH
hash -r

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# GPU preflight: detect GPU type and select / rebuild the matching venv.
# ---------------------------------------------------------------------------
detect_gpu_type() {
  local gpu_name="unknown"

  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name="$(
      nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
        | head -n1 \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's/[^a-z0-9 ]/ /g' \
        | tr -s ' ' ' ' \
        | sed 's/^ *//; s/ *$//'
    )"
  fi

  case "$gpu_name" in
    *h200*) echo "h200|$gpu_name" ;;
    *h100*|*gh200*) echo "h100|$gpu_name" ;;
    *a100*|*l40*|*l40s*|*a10g*|*"rtx 6000 ada"*) echo "a100|$gpu_name" ;;
    *) echo "a100|$gpu_name" ;;
  esac
}

log "Detecting GPU for venv selection"
GPU_INFO="$(detect_gpu_type)"
GPU_TYPE="${GPU_INFO%%|*}"
GPU_NAME="${GPU_INFO#*|}"

log "Detected GPU: ${GPU_NAME:-unknown} -> using venv for $GPU_TYPE"

# If the user did not explicitly set VENV_DIR, switch to the GPU-specific venv.
if [[ "$VENV_DIR" == "${REPO_ROOT}/.venv" || -z "${VENV_DIR:-}" ]]; then
  VENV_DIR="${REPO_ROOT}/.venv-${GPU_TYPE}"
fi

BUILD_SIG_FILE="$VENV_DIR/.vllm-build-signature"
NEED_REBUILD=0

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  NEED_REBUILD=1
elif [[ ! -f "$BUILD_SIG_FILE" ]]; then
  NEED_REBUILD=1
elif ! grep -q "\"gpu_type\": \"${GPU_TYPE}\"" "$BUILD_SIG_FILE" 2>/dev/null; then
  NEED_REBUILD=1
fi

if [[ "$NEED_REBUILD" == "1" ]]; then
  log "Building/rebuilding venv for $GPU_TYPE at $VENV_DIR"
  VENV_DIR="$VENV_DIR" FORCE_REBUILD=1 SETUP_SKIP_GIT_FETCH=1 bash setup_env.sh
fi

# ---------------------------------------------------------------------------
# Activate venv
# ---------------------------------------------------------------------------
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: venv not found at $VENV_DIR. Create it with:" >&2
  echo "  VENV_DIR=$VENV_DIR bash setup_env.sh" >&2
  exit 1
fi

log "Activating venv: $VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

printf '{"gpu_type": "%s", "gpu_name": "%s"}\n' "$GPU_TYPE" "$GPU_NAME" > "$BUILD_SIG_FILE"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log "Python: $(python --version) at $(command -v python)"
log "vllm:   $(command -v vllm || echo not-found)"

# ---------------------------------------------------------------------------
# Runtime configuration (sizing already resolved from CLI / profile / GPUs)
# ---------------------------------------------------------------------------
SERVED_NAME="${SERVED_NAME_OPT:-${SERVED_NAME:-qwen-agent}}"
PORT="${PORT_OPT:-${PORT:-8001}}"
DTYPE="${DTYPE:-bfloat16}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"
export VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_READY_TIMEOUT"

# Matched-compile-off config from the known-good baseline run.
VLLM_COMPILATION_CONFIG='{"mode":0,"custom_ops":["none"],"pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false}}'

# Why we keep --enforce-eager:
#   We pin compilation_config.mode=0 (NO_COMPILATION) for stability. vLLM 0.18
#   then auto-overrides cudagraph_mode -> NONE anyway, so dropping
#   --enforce-eager gains zero speed but routes through a different warmup
#   path (_dummy_run) that hits cudaErrorNoKernelImageForDevice in FlashAttn
#   on cu130 builds. Net: eager is strictly the safe and equivalent choice.
#   Set ENFORCE_EAGER=0 to opt back into the experimental fast rung.
PRIMARY_IMPL="${MODEL_IMPL:-auto}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# Fallback rungs (smaller context / transformers backend) on top of the
# tier-derived primary. These are tier-agnostic — vLLM only cares about
# (max_model_len, gpu_memory_utilization, model-impl) being something it
# can satisfy on the assigned device(s).
FALLBACK1_IMPL="auto"
# Fallback rung 1: step down to 16K to recover from OOM at 32K.
FALLBACK1_MAX_LEN=$(( PRIMARY_MAX_LEN > 16384 ? 16384 : (PRIMARY_MAX_LEN * 3 / 4) ))
FALLBACK1_MEM_UTIL="$(awk -v u="$PRIMARY_MEM_UTIL" 'BEGIN{printf "%.2f", u-0.05}')"
FALLBACK2_IMPL="transformers"
# Fallback rung 2: minimal 8K context, HF transformers backend — last resort.
FALLBACK2_MAX_LEN=$(( PRIMARY_MAX_LEN > 8192 ? 8192 : PRIMARY_MAX_LEN ))
FALLBACK2_MEM_UTIL="$(awk -v u="$PRIMARY_MEM_UTIL" 'BEGIN{printf "%.2f", u-0.15}')"

# Truncation budgets — sized for a 32 K-token model context. _shrink_messages
# subtracts the tools JSON length from the per-message budgets at runtime, so
# the values here are the *base* content budget (excluding tools). If vLLM
# falls back to a smaller rung (16K/8K), the patch's iterative shrink loop
# will tighten these automatically.
#   SOFT = warn + trim oldest turns at this char count (~24K chars ≈ 32K tokens).
#   HARD = hard-truncate remaining content to fit — keeps the last few turns intact.
export TAU_PATCH_MAX_TOOL_CHARS="${TAU_PATCH_MAX_TOOL_CHARS:-2000}"
export TAU_PATCH_SOFT_BUDGET="${TAU_PATCH_SOFT_BUDGET:-40000}"
export TAU_PATCH_HARD_BUDGET="${TAU_PATCH_HARD_BUDGET:-24000}"

# Wall-clock notes (per controller × benchmark cell, eager BF16):
#   1×A100 / 7B  : ~22 LLM calls/τ-traj × 30 tasks × 3 trials, conc=4 → ~1.5 h/cell
#   2×A100 / 32B : ~22 calls × 30 × 3, TP=2, conc=8 → ~1.0–1.5 h/cell
#   12 cells total + BFCL + summary slack → fits in the 10–15 h budget.
#
# These are static knobs that the profile / CLI never override.
# IMPORTANT: use "test" for valid comparison — the train split has 87% 1-2 action
# tasks (trivially easy) vs the test split's mean of 5.1 actions up to 14.
# Running evaluation on train inflates ALL agents uniformly and is not a
# meaningful benchmark (confirmed: retail train=500 tasks avg 1.54 actions,
# retail test=115 tasks avg 5.06 actions).
TAU_TASK_SPLIT="${TAU_TASK_SPLIT:-test}"
TAU_START_INDEX="${TAU_START_INDEX:-0}"
TAU_TEMPERATURE="${TAU_TEMPERATURE:-0.0}"
BFCL_CATEGORIES="${BFCL_CATEGORIES_OPT:-${BFCL_CATEGORIES:-simple,multiple,parallel,parallel_multiple,multi_turn_base}}"

# ---------------------------------------------------------------------------
# Internal train/test splits for benchmarks that lack an upstream split.
#
# These exist purely to keep PRAXIS evaluation honest: PRAXIS pre-warm runs
# on the "train" indices (which feed the experience bank); the held-out
# "test" indices are evaluated by all four controllers with promotion gated
# off, so PRAXIS cannot learn from its own eval set.
#
#   AIRLINE_TRAIN_END     — first N tau-bench airline task indices reserved as
#                           PRAXIS train pool (airline ships only a "test"
#                           split upstream). Test eval runs on
#                           [AIRLINE_TRAIN_END, end-index). Default 20.
#   BFCL_TRAIN_FRACTION   — per-category fraction of BFCL tasks used as the
#                           internal train pool. Default 0.25 (25%).
#
# Override via env vars or the corresponding CLI flags below. Set
# AIRLINE_TRAIN_END=0 / BFCL_TRAIN_FRACTION=0 to disable the split (this
# reverts to legacy behaviour and is NOT recommended for paper-grade runs).
AIRLINE_TRAIN_END="${AIRLINE_TRAIN_END:-20}"
BFCL_TRAIN_FRACTION="${BFCL_TRAIN_FRACTION:-0.25}"

# OpenAI SDK timeout for direct in-process calls.
# At 32K context, prefill of a long trajectory can be substantial; timeouts
# are raised from the 12K-era defaults to accommodate longer sequences.
#   7B  on 1×A100 (32K): 90 s (was 60 s at 12K).
#   14B on 1×A100 (32K): 120 s (was 90 s at 12K; still ~1.8–2.0× slower than 7B).
#   32B on TP=2   (32K): 150 s (was 120 s at 12K).
if [[ "$MODEL_TIER" == "32b" ]]; then
  export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-150}"
elif [[ "$MODEL_TIER" == "14b" ]]; then
  export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
else
  export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-90}"
fi

OUTPUTS_DIR="${OUTPUTS_DIR:-${REPO_ROOT}/outputs}"
mkdir -p "$OUTPUTS_DIR/summary"

# ---------------------------------------------------------------------------
# Resolved-config dump (also runs as the body of --dry-run).
# ---------------------------------------------------------------------------
print_resolved_config() {
  cat <<INFO
---------- Resolved configuration ----------
GPUS=$GPUS_CSV   NUM_GPUS=$NUM_GPUS   TP=$TP
MODEL_TIER=$MODEL_TIER   primary_max_len=$PRIMARY_MAX_LEN   gpu_mem_util=$PRIMARY_MEM_UTIL
PROFILE=$PROFILE
  TAU:  tasks=[$TAU_START_INDEX..$TAU_END_INDEX) trials=$TAU_NUM_TRIALS conc=$TAU_MAX_CONCURRENCY max_steps=$TAU_MAX_STEPS
  BFCL: limit=$BFCL_LIMIT conc=$BFCL_MAX_CONCURRENCY max_steps=$BFCL_MAX_STEPS data_dir=$BFCL_DATA_DIR
CONTROLLERS=$CONTROLLERS_OPT  SKIP_BFCL=$SKIP_BFCL  TAU_ONLY=${TAU_ONLY:-(both)}
PORT=$PORT  SERVED_NAME=$SERVED_NAME  ENFORCE_EAGER=$ENFORCE_EAGER
MODEL_CANDIDATES:
$(printf '  - %s\n' "${MODEL_CANDIDATES[@]}")
--------------------------------------------
INFO
}
print_resolved_config

for cmd in curl lsof vllm python; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: missing command: $cmd" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------------
VLLM_PID=""
ACTIVE_MODEL=""
ACTIVE_IMPL=""
ACTIVE_MAX_LEN=""

wait_health () {
  # Usage: wait_health <url> <timeout_s> [pid]
  # Polls <url> until it returns 200 or <timeout_s> seconds elapse.
  # If the optional <pid> is supplied, also exits early (return 1) if the
  # server process dies before the health endpoint responds — this makes
  # failed rungs bail out in seconds rather than waiting the full timeout.
  local url="$1" timeout="$2" pid="${3:-}" slept=0 step=3
  while (( slept < timeout )); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    # Early-exit: if the server PID died, there is no point continuing.
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      sleep 3  # let any final log lines flush
      return 1
    fi
    sleep "$step"; slept=$(( slept + step ))
  done
  return 1
}

kill_port () {
  local pids; pids="$(lsof -ti tcp:"$1" 2>/dev/null || true)"
  [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
}

kill_vllm () {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  kill_port "$PORT"
  VLLM_PID=""
}
trap 'kill_vllm' EXIT

start_vllm_once () {
  # Usage: start_vllm_once <model> <impl> <max_len> <mem_util> <eager> <label> [attn_backend]
  #
  # Optional 7th arg: VLLM_ATTENTION_BACKEND value to inject into the server
  # environment. Use "TORCH_SDPA" when FlashAttention fails with
  # cudaErrorNoKernelImageForDevice (GPU arch mismatch) — this forces PyTorch's
  # built-in scaled_dot_product_attention, which works on any GPU.
  local model="$1" impl="$2" max_len="$3" mem_util="$4" eager="$5" label="$6"
  local attn_backend="${7:-}"
  local log_file="$OUTPUTS_DIR/vllm.log"
  kill_vllm
  rm -f "$log_file"
  # `set -u` makes empty arrays brittle in older bash. Use a plain string flag.
  local eager_flag=""
  if [[ "$eager" == "1" ]]; then
    eager_flag="--enforce-eager"
  fi
  log "Starting vLLM [$label] model=$model impl=$impl max_len=$max_len mem_util=$mem_util TP=$TP gpus=$GPUS_CSV eager=$eager${attn_backend:+ attn=$attn_backend}"
  # NOTE: $eager_flag is intentionally unquoted so the empty case expands to
  #       nothing (vllm sees no extra arg). When set, it's a single flag with
  #       no spaces, so word-splitting is safe.
  #
  # VLLM_ATTENTION_BACKEND is injected only when attn_backend is non-empty,
  # using the `env VAR=val cmd` prefix form which limits the variable's scope
  # to the subprocess (avoids polluting the shell for subsequent rungs).
  if [[ -n "$attn_backend" ]]; then
    CUDA_VISIBLE_DEVICES="$GPUS_CSV" VLLM_ATTENTION_BACKEND="$attn_backend" \
      vllm serve "$model" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --model-impl "$impl" \
        --dtype "$DTYPE" \
        --max-model-len "$max_len" \
        --gpu-memory-utilization "$mem_util" \
        --tensor-parallel-size "$TP" \
        --enable-auto-tool-choice \
        --tool-call-parser "$TOOL_CALL_PARSER" \
        --disable-log-stats \
        $eager_flag \
        --compilation-config "$VLLM_COMPILATION_CONFIG" \
        > "$log_file" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$GPUS_CSV" \
      vllm serve "$model" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --model-impl "$impl" \
        --dtype "$DTYPE" \
        --max-model-len "$max_len" \
        --gpu-memory-utilization "$mem_util" \
        --tensor-parallel-size "$TP" \
        --enable-auto-tool-choice \
        --tool-call-parser "$TOOL_CALL_PARSER" \
        --disable-log-stats \
        $eager_flag \
        --compilation-config "$VLLM_COMPILATION_CONFIG" \
        > "$log_file" 2>&1 &
  fi
  VLLM_PID=$!

  # Pass VLLM_PID so wait_health can bail early if the process crashes
  # (e.g. cudaErrorNoKernelImageForDevice) rather than waiting the full timeout.
  if wait_health "http://127.0.0.1:${PORT}/health" "$VLLM_READY_TIMEOUT" "$VLLM_PID"; then
    log "vLLM healthy [$label] with $model"
    ACTIVE_MODEL="$model"; ACTIVE_IMPL="$impl"; ACTIVE_MAX_LEN="$max_len"
    printf '%s\n' "$model" > "$OUTPUTS_DIR/active_model.txt"
    return 0
  fi
  log "vLLM failed [$label]; tail of log:"
  tail -n 80 "$log_file" >&2 || true
  kill_vllm
  return 1
}

# ---------------------------------------------------------------------------
# Apply vLLM thread-safety patch to fix "RuntimeError: Already borrowed".
#
# Root cause: vLLM creates a new tool-parser instance per request, and each
# instantiation calls tokenizer.encode(), which clashes with the async
# engine's internal tokenizer borrow (Rust RefCell panic). The patch wraps
# every encode() call in a threading.Lock + retry, and adds a retry loop
# around tool_parser_cls(tokenizer) in serving.py.  Idempotent — safe to
# re-run across subsequent invocations of run_project.sh.
# ---------------------------------------------------------------------------
_VLLM_SRC_DIR=""
for _d in \
    "${EXTERNAL_DIR}/vllm-src-0.18.0" \
    "${EXTERNAL_DIR}/vllm-src" \
    "${EXTERNAL_DIR}/vllm" \
    "${REPO_ROOT}/external/vllm-src-0.18.0" \
    "${REPO_ROOT}/external/vllm-src"; do
  if [[ -d "$_d/vllm/tool_parsers" ]]; then
    _VLLM_SRC_DIR="$_d"; break
  fi
done
if [[ -n "$_VLLM_SRC_DIR" ]]; then
  log "Applying vLLM tokenizer-borrow patch: $_VLLM_SRC_DIR"
  python "$REPO_ROOT/src/_vllm_patches/fix_tokenizer_borrow.py" "$_VLLM_SRC_DIR" \
    || log "WARNING: vLLM patch failed — will attempt run anyway"
else
  log "WARNING: vLLM source directory not found; skipping tokenizer-borrow patch"
fi

# ---------------------------------------------------------------------------
# Try (model candidate × launch config) combos until one serves.
# Per-model rungs:
#   1. fast:   CUDA graphs ON, primary context  (skipped if ENFORCE_EAGER=1)
#   2. safe:   eager,         primary context  (previous known-good config)
#   3. small:  eager,         12K context
#   4. sdpa:   eager, TORCH_SDPA backend, primary context
#              — forces PyTorch's built-in scaled_dot_product_attention,
#              bypassing FlashAttn v2 custom CUDA kernels.  Use this when
#              the cluster scheduler assigns a GPU with a compute capability
#              the bundled FlashAttn wheels were not compiled for
#              (cudaErrorNoKernelImageForDevice during _dummy_run warmup).
#   5. cpu-ish: transformers,  8K  context
#              — full HuggingFace transformers model runner; uses SDPA via
#              the transformers library.  Slowest but works on any GPU.
# ---------------------------------------------------------------------------
STARTED=0
for m in "${MODEL_CANDIDATES[@]}"; do
  if [[ "$ENFORCE_EAGER" != "1" ]]; then
    if start_vllm_once "$m" "$PRIMARY_IMPL"   "$PRIMARY_MAX_LEN"   "$PRIMARY_MEM_UTIL"   "0" "fast/$m";          then STARTED=1; break; fi
  fi
  if start_vllm_once   "$m" "$PRIMARY_IMPL"   "$PRIMARY_MAX_LEN"   "$PRIMARY_MEM_UTIL"   "1" "safe/$m";          then STARTED=1; break; fi
  if start_vllm_once   "$m" "$FALLBACK1_IMPL" "$FALLBACK1_MAX_LEN" "$FALLBACK1_MEM_UTIL" "1" "small/$m";         then STARTED=1; break; fi
  if start_vllm_once   "$m" "$PRIMARY_IMPL"   "$PRIMARY_MAX_LEN"   "$PRIMARY_MEM_UTIL"   "1" "sdpa/$m"   "TORCH_SDPA"; then STARTED=1; break; fi
  if start_vllm_once   "$m" "$FALLBACK2_IMPL" "$FALLBACK2_MAX_LEN" "$FALLBACK2_MEM_UTIL" "1" "transformers/$m";  then STARTED=1; break; fi
  log "All configs failed for $m; trying next candidate"
done

if [[ "$STARTED" != "1" ]]; then
  log "ERROR: no model candidate could be served"
  exit 1
fi

log "Using $ACTIVE_MODEL (impl=$ACTIVE_IMPL max_len=$ACTIVE_MAX_LEN) for ALL controllers"

# ---------------------------------------------------------------------------
# OpenAI-compatible client config
# ---------------------------------------------------------------------------
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_BASE="$OPENAI_BASE_URL"

# Runtime memory accumulates lessons distilled from saved trajectories across
# runs. Lives under PROJECT_SCRATCH so cards survive cluster job boundaries and
# are available to every re-invocation of this script on the same scratch volume.
export REX_RUNTIME_DIR="${REX_RUNTIME_DIR:-${PROJECT_SCRATCH}/rex_experience_runtime}"
mkdir -p "$REX_RUNTIME_DIR"
# No separate pre-built seed bank exists on the scratch volume — only
# rex_experience_runtime is available. Seed bank defaults to the same directory
# so REx loads all accumulated experience cards from rex_experience_runtime.
# MemoryStore deduplicates by signature so loading the same card twice is safe.
export REX_EXPERIENCE_DIR="${REX_EXPERIENCE_DIR:-${REX_RUNTIME_DIR}}"
mkdir -p "$REX_EXPERIENCE_DIR"
# Stateful re-retrieval cadence: refresh the playbook every N effective steps.
# 0 disables (only the initial brief is used).
export REX_RETRIEVAL_REFRESH_EVERY="${REX_RETRIEVAL_REFRESH_EVERY:-2}"
if [[ "$CONTROLLERS_OPT" == *"praxis"* || "$CONTROLLERS_OPT" == *"rex"* ]]; then
  log "Using PRAXIS experience bank:          $REX_EXPERIENCE_DIR"
  log "Using PRAXIS runtime memory directory: $REX_RUNTIME_DIR"
  # NOTE: we do NOT auto-generate a seed bank here.
  # PRAXIS starts with whatever experience cards already exist in REX_EXPERIENCE_DIR.
  # On the very first run that directory is empty — this is intentional.
  # To build permanent experience memory from collected trajectories, run:
  #
  #   python -m src.scripts.build_experience_memory \
  #       --inputs "$OUTPUTS_DIR" --output-bank "$REX_EXPERIENCE_DIR"
fi

# ---------------------------------------------------------------------------
# NOTE: raw trajectories are NEVER automatically re-scanned at runtime.
# Permanent experience memory is built ONLY by an explicit offline step:
#
#   python -m src.scripts.build_experience_memory \
#       --inputs "$OUTPUTS_DIR" --output-bank "$REX_EXPERIENCE_DIR"
#
# Run that script once after collecting baseline/act/react trajectories,
# then re-run run_project.sh so REx retrieves from the permanent memory.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
run_tau () {
  local env_name="$1" agent_kind="$2"
  local out="$OUTPUTS_DIR/tau_${env_name}_${agent_kind}"
  mkdir -p "$out"
  log "tau-bench: env=$env_name agent=$agent_kind -> $out"

  # Split selection:
  #   retail  — uses TAU_TASK_SPLIT (default "test"; 115 tasks avg 5.1 actions).
  #             DO NOT use "train": it has 500 trivial tasks (87% need ≤2 calls)
  #             that inflate every agent's score uniformly and is not a valid eval.
  #   airline — only ships a "test" split; "train" does not exist.
  local split_arg="$TAU_TASK_SPLIT"
  if [[ "$env_name" == "airline" ]]; then
    split_arg="test"
  fi

  # Promotion policy — only PRAXIS should write to the experience bank:
  #   • baseline / act / react → "never": these agents have no procedural memory
  #     and their trajectories would contaminate the PRAXIS experience bank with
  #     non-memory-augmented patterns.
  #   • praxis on retail (eval = upstream test split) → "auto": _should_promote()
  #     returns False for "test" split, so no promotion during eval. Pre-warm
  #     the bank from the upstream "train" split (see PRAXIS_PREWARM below).
  #   • praxis on airline → "auto": we use an internal split (AIRLINE_TRAIN_END)
  #     because airline ships only a "test" split upstream. The pipeline's
  #     per-record _is_test_split_record gate keeps eval-range records out of
  #     the bank even if promotion is invoked.
  local promote_arg="never"
  if [[ "$agent_kind" == "rex" || "$agent_kind" == "praxis" ]]; then
    promote_arg="auto"
  fi

  # Airline-specific: evaluate only on the held-out test slice of the upstream
  # "test" split. Train indices [0, AIRLINE_TRAIN_END) are reserved for the
  # optional PRAXIS pre-warm and skipped during eval here.
  local tau_start="$TAU_START_INDEX"
  local internal_end_arg=()
  if [[ "$env_name" == "airline" && "$AIRLINE_TRAIN_END" -gt 0 ]]; then
    if [[ "$TAU_START_INDEX" -lt "$AIRLINE_TRAIN_END" ]]; then
      tau_start="$AIRLINE_TRAIN_END"
    fi
    # Pass the boundary so per-record info.task_split tagging works for any
    # records that *do* get distilled (e.g. during pre-warm reuse of this fn).
    internal_end_arg=(--internal-train-end "$AIRLINE_TRAIN_END")
  fi

  if python -m src.runners.tau_runner \
      --env "$env_name" --agent "$agent_kind" \
      --model "$SERVED_NAME" --user-model "$SERVED_NAME" \
      --task-split "$split_arg" \
      --promote-runtime-memory "$promote_arg" \
      --start-index "$tau_start" --end-index "$TAU_END_INDEX" \
      "${internal_end_arg[@]}" \
      --num-trials "$TAU_NUM_TRIALS" \
      --max-concurrency "$TAU_MAX_CONCURRENCY" \
      --temperature "$TAU_TEMPERATURE" \
      --max-num-steps "$TAU_MAX_STEPS" \
      --runtime-dir "$REX_RUNTIME_DIR" \
      --output-dir "$out"; then
    log "tau-bench OK: $env_name/$agent_kind"
  else
    log "WARNING: tau-bench FAILED: $env_name/$agent_kind (continuing)"
  fi
}

run_bfcl () {
  local agent_kind="$1"
  local out="$OUTPUTS_DIR/bfcl_agent_${agent_kind}"
  mkdir -p "$out"
  log "BFCL V4: agent=$agent_kind -> $out"

  # Auto-download BFCL data if the directory doesn't exist or is empty.
  if [[ ! -d "$BFCL_DATA_DIR" ]] || [[ -z "$(ls -A "$BFCL_DATA_DIR" 2>/dev/null)" ]]; then
    log "BFCL data not found at $BFCL_DATA_DIR — downloading..."
    if python -m src.scripts.download_bfcl --output-dir "$BFCL_DATA_DIR"; then
      log "BFCL download OK"
    else
      log "WARNING: BFCL download failed — BFCL cells may be skipped"
    fi
  fi

  # Same promotion policy as tau: only PRAXIS writes to the experience bank.
  # Evaluation runs on the held-out "test" slice of the internal per-category
  # split (BFCL has no upstream train/test distinction). PRAXIS pre-warm
  # (below) runs on the "train" slice. The pipeline's _is_test_split_record
  # gate is the defence-in-depth backstop.
  local bfcl_promote="never"
  [[ "$agent_kind" == "rex" || "$agent_kind" == "praxis" ]] && bfcl_promote="auto"

  if python -m src.runners.bfcl_runner \
      --agent "$agent_kind" --model "$SERVED_NAME" \
      --data-dir "$BFCL_DATA_DIR" \
      --categories "$BFCL_CATEGORIES" \
      --limit "$BFCL_LIMIT" \
      --max-num-steps "$BFCL_MAX_STEPS" \
      --max-concurrency "$BFCL_MAX_CONCURRENCY" \
      --promote-runtime-memory "$bfcl_promote" \
      --train-fraction "$BFCL_TRAIN_FRACTION" \
      --task-split test \
      --runtime-dir "$REX_RUNTIME_DIR" \
      --output-dir "$out"; then
    log "BFCL V4 OK: $agent_kind"
  else
    log "WARNING: BFCL V4 FAILED: $agent_kind (continuing)"
  fi
}

# ---------------------------------------------------------------------------
# Optional PRAXIS pre-warm: run praxis on the retail TRAIN split to seed the
# experience bank BEFORE the main evaluation pass on the TEST split.
#
# Why: the test split tasks average 5.1 actions; with _should_promote()
# blocking promotion on "test", PRAXIS starts cold unless we pre-warm here.
# Without pre-warm, results are still valid (cold-start baseline) but weaker
# than what the system can do with accumulated experience.
#
# Enable with --praxis-prewarm or PRAXIS_PREWARM=1.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Phase B donor helpers — run a single non-PRAXIS controller on the TRAIN
# slice of a benchmark and distill its trajectories into the shared
# experience bank. These are only used during Phase B (experience harvest);
# Phase A and Phase C use the standard run_tau / run_bfcl functions.
# ---------------------------------------------------------------------------
run_tau_donor () {
  local env_name="$1" donor="$2"
  local out="$OUTPUTS_DIR/donor_${env_name}_${donor}"
  mkdir -p "$out"
  log "Phase B donor: env=$env_name agent=$donor -> $out"

  local split_arg internal_end_arg=() start_idx end_idx
  if [[ "$env_name" == "retail" ]]; then
    split_arg="train"
    start_idx=0
    end_idx="$TAU_END_INDEX"
  else  # airline
    split_arg="test"
    internal_end_arg=(--internal-train-end "$AIRLINE_TRAIN_END")
    start_idx=0
    end_idx="$AIRLINE_TRAIN_END"
    if [[ "$end_idx" -le 0 ]]; then
      log "Phase B donor: AIRLINE_TRAIN_END<=0; skipping $env_name/$donor"
      return 0
    fi
  fi

  if python -m src.runners.tau_runner \
      --env "$env_name" --agent "$donor" \
      --model "$SERVED_NAME" --user-model "$SERVED_NAME" \
      --task-split "$split_arg" \
      --promote-runtime-memory auto \
      --start-index "$start_idx" --end-index "$end_idx" \
      "${internal_end_arg[@]}" \
      --num-trials 1 \
      --max-concurrency "$TAU_MAX_CONCURRENCY" \
      --temperature "$TAU_TEMPERATURE" \
      --max-num-steps "$TAU_MAX_STEPS" \
      --runtime-dir "$REX_RUNTIME_DIR" \
      --output-dir "$out"; then
    log "Phase B donor OK: $env_name/$donor"
  else
    log "WARNING: Phase B donor FAILED: $env_name/$donor (continuing)"
  fi
}

run_bfcl_donor () {
  local donor="$1"
  local out="$OUTPUTS_DIR/donor_bfcl_${donor}"
  mkdir -p "$out"
  log "Phase B donor: bfcl agent=$donor -> $out"
  if python -m src.runners.bfcl_runner \
      --agent "$donor" --model "$SERVED_NAME" \
      --data-dir "$BFCL_DATA_DIR" \
      --categories "$BFCL_CATEGORIES" \
      --limit "$BFCL_LIMIT" \
      --max-num-steps "$BFCL_MAX_STEPS" \
      --max-concurrency "$BFCL_MAX_CONCURRENCY" \
      --promote-runtime-memory auto \
      --train-fraction "$BFCL_TRAIN_FRACTION" \
      --task-split train \
      --runtime-dir "$REX_RUNTIME_DIR" \
      --output-dir "$out"; then
    log "Phase B donor OK: bfcl/$donor"
  else
    log "WARNING: Phase B donor FAILED: bfcl/$donor (continuing)"
  fi
}

# ---------------------------------------------------------------------------
# Tau-bench env list (shared by Phase A and Phase C).
# ---------------------------------------------------------------------------
case "${TAU_ONLY:-}" in
  ""|"both") TAU_ENVS=("retail" "airline") ;;
  retail)    TAU_ENVS=("retail") ;;
  airline)   TAU_ENVS=("airline") ;;
esac

# Comma-split helpers.
IFS=',' read -r -a EVAL_LIST  <<<"$EVAL_CONTROLLERS"
IFS=',' read -r -a DONOR_LIST <<<"$DONOR_CONTROLLERS"

# ===========================================================================
# PHASE A — Baseline evaluation (CLEAN, no recording)
# ===========================================================================
# Every controller in $EVAL_CONTROLLERS runs on the held-out TEST split.
# run_tau / run_bfcl set --promote-runtime-memory=never for non-PRAXIS agents,
# so these trajectories are scored and discarded. PRAXIS is intentionally
# NOT in this list — its evaluation happens in Phase C after the bank has
# been populated.
# ---------------------------------------------------------------------------
if [[ "$RUN_PHASE_A" == "1" ]]; then
  log "=================================================================="
  log "PHASE A — Baseline evaluation on TEST (no recording)"
  log "  controllers: $EVAL_CONTROLLERS"
  log "=================================================================="
  for env_name in "${TAU_ENVS[@]}"; do
    for c in "${EVAL_LIST[@]}"; do
      c="$(echo "$c" | tr -d '[:space:]')"
      [[ -z "$c" ]] && continue
      [[ "$c" == "praxis" || "$c" == "rex" ]] && continue   # Phase C only
      run_tau "$env_name" "$c"
    done
  done
  if [[ "$SKIP_BFCL" != "1" ]]; then
    for c in "${EVAL_LIST[@]}"; do
      c="$(echo "$c" | tr -d '[:space:]')"
      [[ -z "$c" ]] && continue
      [[ "$c" == "praxis" || "$c" == "rex" ]] && continue
      run_bfcl "$c"
    done
  else
    log "Phase A: skipping BFCL cells (--skip-bfcl)."
  fi
else
  log "Phase A skipped (RUN_PHASE_A=0)."
fi

# ===========================================================================
# PHASE B — Experience harvest from a diverse donor cohort
# ===========================================================================
# Every controller in $DONOR_CONTROLLERS runs on the TRAIN slice with
# --promote-runtime-memory=auto. Cards from each donor accumulate in
# $REX_RUNTIME_DIR. The diversity of prompting priors (act, react, cot,
# plan-solve, reflexion-lite, plus PRAXIS itself for self-experience)
# widens the procedural-memory pool beyond what a single baseline can
# provide. Run-level and record-level gates keep TEST records out.
# ---------------------------------------------------------------------------
if [[ "$RUN_PHASE_B" == "1" ]]; then
  # Optional clean slate so the donor harvest writes into an empty bank.
  if [[ "$PRAXIS_RESET_BANK" == "1" ]]; then
    log "PHASE B: PRAXIS_RESET_BANK=1 — clearing $REX_RUNTIME_DIR"
    rm -rf "$REX_RUNTIME_DIR" && mkdir -p "$REX_RUNTIME_DIR"
  fi

  log "=================================================================="
  log "PHASE B — Donor-cohort experience harvest on TRAIN"
  log "  donors: $DONOR_CONTROLLERS"
  log "  bank  : $REX_RUNTIME_DIR"
  log "=================================================================="
  # Tau-bench donors
  for env_name in "${TAU_ENVS[@]}"; do
    for d in "${DONOR_LIST[@]}"; do
      d="$(echo "$d" | tr -d '[:space:]')"
      [[ -z "$d" ]] && continue
      [[ "$d" == "baseline" ]] && d="baseline"  # alias kept for clarity
      run_tau_donor "$env_name" "$d"
    done
  done
  # BFCL donors
  _BFCL_HAS_TRAIN="$(awk -v x="$BFCL_TRAIN_FRACTION" 'BEGIN{print (x>0)?1:0}')"
  if [[ "$SKIP_BFCL" != "1" && "$_BFCL_HAS_TRAIN" == "1" ]]; then
    for d in "${DONOR_LIST[@]}"; do
      d="$(echo "$d" | tr -d '[:space:]')"
      [[ -z "$d" ]] && continue
      run_bfcl_donor "$d"
    done
  else
    log "Phase B: skipping BFCL donors (skip-bfcl or BFCL_TRAIN_FRACTION=0)."
  fi
else
  log "Phase B skipped (RUN_PHASE_B=0) — PRAXIS will evaluate cold-start."
fi

# ===========================================================================
# PHASE C — PRAXIS evaluation on TEST (warm bank from Phase B)
# ===========================================================================
# With the bank populated by Phase B, PRAXIS now runs on the held-out TEST
# split. The promotion gate stays on, so PRAXIS cannot learn from its own
# eval; any train-slice trajectories it produces (in mixed airline runs)
# are still eligible. Results are reported alongside Phase A baselines.
# ---------------------------------------------------------------------------
if [[ "$RUN_PHASE_C" == "1" ]]; then
  log "=================================================================="
  log "PHASE C — PRAXIS evaluation on TEST (warm bank)"
  log "=================================================================="
  for env_name in "${TAU_ENVS[@]}"; do
    run_tau "$env_name" "praxis"
  done
  if [[ "$SKIP_BFCL" != "1" ]]; then
    run_bfcl "praxis"
  fi
else
  log "Phase C skipped (RUN_PHASE_C=0)."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "Stopping vLLM before summary"
kill_vllm

log "Building summary"
python -m src.summary.build_summary \
  --outputs-dir "$OUTPUTS_DIR" \
  --active-model "$ACTIVE_MODEL" \
  --served-name "$SERVED_NAME"

log "Done."
log "  summary.json: $OUTPUTS_DIR/summary/summary.json"
log "  summary.md:   $OUTPUTS_DIR/summary/summary.md"
