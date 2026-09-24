#!/bin/bash
# =============================================================================
# setup_env.sh -- ONE-TIME install of the Isaac stack on VT ARC (login node).
# =============================================================================
#   bash arc/setup_env.sh            # preflight, then install (idempotent)
#   bash arc/setup_env.sh --check    # preflight only
#   bash arc/setup_env.sh --verify   # verify an existing env only
#
# Route: pip-installed Isaac Sim inside a DEDICATED conda env. ARC gives no
# Docker, and pip Isaac Sim needs no root and no container runtime -- it is the
# same "absolute-path conda env" pattern already proven on ARC.
#
# DO NOT reuse or merge with the VLM project's vrr / vrr-train / vrr-gen envs.
# Isaac Sim pins its own torch + numpy<2; crossing envs fails at import, late.
#
# Version pairing (must move together -- a mismatched pair fails at import):
#   Isaac Sim 4.5.0  <->  Isaac Lab v2.1.0  <->  Python 3.10  <->  torch 2.5.1/cu121
# The codebase imports the Isaac Lab 2.x namespace (isaaclab.*, isaaclab_rl.*,
# isaaclab_assets.*). Isaac Lab 1.x uses omni.isaac.lab.* and will NOT work.
# VERIFY ON ARC: the pairing against Isaac Lab's installation docs for the tag.
# =============================================================================
set -euo pipefail

RTN_ENV="${RTN_ENV:-$HOME/miniconda3/envs/rtn}"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
ISAACLAB_REF="v2.1.0"
ISAACSIM_VERSION="4.5.0"
PY_VERSION="3.10"
TORCH_PIN=(torch==2.5.1 torchvision==0.20.1)
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
MIN_GLIBC="2.34"        # pip isaacsim wheels are manylinux_2_34
MIN_FREE_GB=60          # env ~30-35 GB + kit caches + checkpoints + headroom

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$RTN_ENV/bin/python"
MODE="${1:-install}"

say()  { echo -e "\n[setup] $*"; }
die()  { echo "[setup] FATAL: $*" >&2; exit 2; }
vge()  { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }  # $1 >= $2

# -----------------------------------------------------------------------------
preflight() {
  say "PREFLIGHT"
  local glibc
  glibc="$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)"
  echo "  glibc        : $glibc (need >= $MIN_GLIBC)"
  if ! vge "$glibc" "$MIN_GLIBC"; then
    die "glibc $glibc is too old for pip Isaac Sim. Use the container route instead:
       Isaac Lab's 'Cluster Guide' (docker/cluster/) -- build the image on a machine
       with Docker, convert with 'apptainer build', copy the .sif to ARC, and run
       the launchers' python via 'apptainer exec --nv'. See setup_notes.md Part 1.3."
  fi
  echo "  compute OS   : $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
  echo "  NOTE: login and compute nodes can differ -- the smoke test re-checks on a GPU node."

  [[ -x "$CONDA_BIN" ]] || die "no conda at $CONDA_BIN (set CONDA_BIN=...)"
  echo "  conda        : $CONDA_BIN"

  local free_gb
  free_gb="$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1024/1024)}')"
  # /home is a SHARED filesystem (hundreds of TB free), so df says nothing
  # about YOUR headroom -- the per-user quota below is the real limit.
  echo "  /home fs free: ${free_gb} GB (shared by all users -- NOT your quota)"
  [[ "$free_gb" -ge "$MIN_FREE_GB" ]] || die "the /home filesystem itself is nearly full"
  echo "  YOUR QUOTA (need >= ${MIN_FREE_GB} GB free on the \$HOME row):"
  (quota 2>/dev/null || echo "    (quota command unavailable)") | sed 's/^/    /'
  if [[ -t 0 ]]; then
    read -r -p "  Does the \$HOME quota show >= ${MIN_FREE_GB} GB free? [y/N] " ok
    [[ "$ok" == [yY]* ]] || die "free up quota first (or set RTN_ENV to a location with room)"
  fi

  command -v git >/dev/null || die "git not found"
  echo "  preflight OK"
}

# -----------------------------------------------------------------------------
install() {
  # Big wheels (isaacsim ~20 GB) overflow a small login-node /tmp mid-install.
  export TMPDIR="$HOME/tmp/rtn-pip"
  mkdir -p "$TMPDIR"
  local PIP=("$PY" -m pip install --no-cache-dir)

  if [[ ! -x "$PY" ]]; then
    say "creating env $RTN_ENV (python $PY_VERSION)"
    "$CONDA_BIN" create -y -p "$RTN_ENV" "python=$PY_VERSION"
  else
    say "env exists: $RTN_ENV"
  fi
  "$PY" -V | grep -q "Python $PY_VERSION" || die "$PY is not Python $PY_VERSION"
  "$PY" -m pip install --upgrade pip

  say "torch (the build Isaac Sim $ISAACSIM_VERSION expects -- install BEFORE isaacsim)"
  "${PIP[@]}" "${TORCH_PIN[@]}" --index-url "$TORCH_INDEX"

  say "Isaac Sim $ISAACSIM_VERSION (pip; extscache avoids extension downloads on compute nodes)"
  "${PIP[@]}" "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com

  say "Isaac Lab $ISAACLAB_REF -> $ISAACLAB_DIR"
  if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  fi
  git -C "$ISAACLAB_DIR" fetch --tags --quiet
  git -C "$ISAACLAB_DIR" checkout --quiet "$ISAACLAB_REF"
  # isaaclab.sh resolves python from the active conda env; point it at ours
  # explicitly rather than trusting `conda activate` (silent no-op in batch shells).
  # VERIFY ON ARC: isaaclab.sh picks $CONDA_PREFIX/bin/python (prints it).
  (
    export CONDA_PREFIX="$RTN_ENV" PATH="$RTN_ENV/bin:$PATH"
    cd "$ISAACLAB_DIR" && ./isaaclab.sh --install rsl_rl
  )

  say "project requirements"
  "${PIP[@]}" -r "$REPO_ROOT/requirements.txt"

  # Unpacked wheels are only needed during install; left behind they hold
  # ~20 GB of quota indefinitely.
  rm -rf "$TMPDIR"
}

# -----------------------------------------------------------------------------
verify() {
  say "VERIFY $RTN_ENV"
  [[ -x "$PY" ]] || die "no python at $PY"
  local v; v="$("$PY" -V 2>&1 || true)"
  [[ "$v" == Python* ]] || die "'$PY -V' printed '$v' -- broken interpreter"
  echo "  $v"
  # Versions via metadata only: importing isaaclab/torch.cuda on a login node
  # can fail for GPU reasons that say nothing about the install.
  "$PY" - <<'EOF'
import importlib.metadata as md, importlib.util as u, sys
need = {"isaacsim": "isaacsim", "isaaclab": "isaaclab", "isaaclab_rl": "isaaclab_rl",
        "isaaclab_assets": "isaaclab_assets", "rsl_rl": "rsl-rl-lib", "torch": "torch",
        "numpy": "numpy", "yaml": "PyYAML", "wandb": "wandb", "matplotlib": "matplotlib"}
bad = False
for mod, dist in need.items():
    found = u.find_spec(mod) is not None
    try:
        ver = md.version(dist)
    except md.PackageNotFoundError:
        ver = "?"
    print(f"  {'ok ' if found else 'MISSING'} {mod:<16} {ver}")
    bad |= not found
np_major = int(md.version("numpy").split(".")[0])
if np_major >= 2:
    print("  FAIL numpy >= 2: Isaac Sim 4.x needs numpy 1.x"); bad = True
sys.exit(1 if bad else 0)
EOF
  # A later `pip install` can silently break an earlier package's pins (seen:
  # wandb pulling protobuf 7 under isaaclab_rl's <5). pip only WARNS at install
  # time, so fail here on any conflict involving the Isaac packages.
  say "dependency conflicts (pip check, Isaac packages)"
  local conflicts
  conflicts="$("$PY" -m pip check 2>/dev/null | grep -iE '^(isaac|rsl)' || true)"
  if [[ -n "$conflicts" ]]; then
    echo "$conflicts" | sed 's/^/  /'
    die "dependency conflicts above -- fix before running on a GPU node"
  fi
  echo "  none"

  say "pure-logic tests (no GPU, no Isaac)"
  (cd "$REPO_ROOT" && "$PY" -m pytest tests/ -q)
  say "VERIFY OK. Next: the smoke test -- see setup_notes.md Part 3."
}

case "$MODE" in
  --check)  preflight ;;
  --verify) verify ;;
  install)  preflight; install; verify ;;
  *) die "usage: bash arc/setup_env.sh [--check|--verify]" ;;
esac
