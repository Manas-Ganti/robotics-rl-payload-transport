# shellcheck shell=bash
# =============================================================================
# arc_env.sh -- sourced by every arc/*.slurm launcher. Not run directly.
# =============================================================================
# Every check here exists because the same failure already cost a GPU
# allocation on ARC in another project (see arc_runbook.md):
#   * a job "succeeding" in 2 s because the interpreter was a 0-byte file
#   * `source activate` silently leaving the job in the wrong env
#   * a variable passed as `VAR=x sbatch` that the launcher never read
# So: absolute env path only, no `module load`/`conda activate`, and hard
# asserts that abort with a readable message BEFORE Isaac Sim spends 10 min
# starting up.
# =============================================================================
set -euo pipefail

_die() { echo "[arc_env] FATAL: $*" >&2; exit 2; }

# ---- repo root: sbatch runs a SPOOLED COPY of the .slurm file, so the script's
# own path is useless. SLURM_SUBMIT_DIR is where `sbatch` was run -- which must
# be the repo root (enforced below).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
[[ -f training/train.py && -d configs ]] \
  || _die "submit from the repo root (SLURM_SUBMIT_DIR=$REPO_ROOT has no training/train.py)"

# ---- the ONE env var launchers read. Absolute path, never a bare name.
: "${RTN_ENV:=$HOME/miniconda3/envs/rtn}"
[[ "$RTN_ENV" = /* ]] || _die "RTN_ENV must be an absolute path, got '$RTN_ENV'"
PY="$RTN_ENV/bin/python"
[[ -x "$PY" ]] || _die "no python at $PY -- run arc/setup_env.sh first"
export PATH="$RTN_ENV/bin:$PATH"

# A real CPython always prints its version. No output == a truncated binary
# that bash runs as an empty script, exit 0 (the expensive 2-second-job trap).
_pyver="$("$PY" -V 2>&1 || true)"
[[ "$_pyver" == Python* ]] || _die "$PY -V printed '$_pyver' -- interpreter is broken (0-byte?)"
"$PY" -c "import sys; assert sys.prefix.startswith('$RTN_ENV'), sys.prefix" \
  || _die "python is not running inside $RTN_ENV"
# find_spec locates packages WITHOUT executing them -- importing isaaclab
# submodules before AppLauncher is itself an error.
"$PY" -c "
import importlib.util as u, sys
missing = [m for m in ('isaacsim', 'isaaclab', 'isaaclab_rl', 'rsl_rl', 'yaml', 'numpy', 'torch') if u.find_spec(m) is None]
sys.exit('missing: ' + ', '.join(missing) if missing else 0)" \
  || _die "Isaac stack incomplete in $RTN_ENV -- re-run arc/setup_env.sh"

# ---- Isaac Sim non-interactive consent (first launch blocks on a prompt otherwise)
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---- secrets (WANDB_API_KEY, optional TELEGRAM_*). Outside the repo, chmod 600.
# Reuses the file the VLM project already set up on ARC.
_secrets="${RTN_SECRETS:-$HOME/.config/vrr/secrets.env}"
# shellcheck disable=SC1090
[[ -f "$_secrets" ]] && source "$_secrets"
export WANDB_DIR="${WANDB_DIR:-$HOME/wandb}"
mkdir -p "$WANDB_DIR"

# ---- single-GPU workload (1 x L40S): no NCCL, no multi-rank. Guard against a stray
# multi-GPU request that would idle 7 GPUs and wreck queue priority.
_ngpu="$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' ')"
[[ "$_ngpu" -ge 1 ]] || _die "no GPU visible (nvidia-smi -L empty) -- did you pass --gres?"
[[ "$_ngpu" -eq 1 ]] || echo "[arc_env] WARNING: $_ngpu GPUs visible; this project uses exactly 1"

echo "[arc_env] job=${SLURM_JOB_ID:-local} node=$(hostname) partition=${SLURM_JOB_PARTITION:-?}"
echo "[arc_env] python=$PY ($_pyver)"
echo "[arc_env] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[arc_env] git=$(git rev-parse --short HEAD 2>/dev/null || echo '?')$(git diff --quiet 2>/dev/null || echo ' (DIRTY)')"

# ---- optional Telegram notifications. Every path no-ops without credentials
# and ends in `|| true`: a notification failure must never kill a run.
arc_notify() {
  [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]] || return 0
  curl -s -m 10 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=$1" >/dev/null || true
}
arc_notify_finish() {
  local rc=$?   # capture FIRST, re-return it: the trap must not mask the job's exit code
  local out
  out="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/.*StdOut=//p')"
  local tail_txt=""
  [[ -f "$out" ]] && tail_txt="$(cut -c1-180 "$out" | tail -n 25)"
  arc_notify "[rtn] ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-} exit=$rc on $(hostname)
${tail_txt}"
  return $rc
}
