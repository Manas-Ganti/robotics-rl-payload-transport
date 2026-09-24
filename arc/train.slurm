#!/bin/bash
# =============================================================================
# train.slurm -- one PPO training run (or the Phase 0 smoke test) on 1 GPU.
# =============================================================================
# Submit FROM THE REPO ROOT. Everything after the script name goes straight to
# training/train.py's argparse, so a typo fails loudly instead of being dropped
# (unlike `VAR=x sbatch`, which silently ignores names a launcher never reads).
#
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu \
#          arc/train.slurm --run-name p1_v1 --resume auto
#
# Walltime: the defaults below are sized for backfill (small + short jumps the
# queue). If a run is killed at walltime, resubmit the IDENTICAL line --
# `--resume auto` + a fixed `--run-name` continues from the last model_<N>.pt.
# A timeout skips model_final.pt; evaluate model_<N>.pt in that case.
#
# GPU: L40S by default. It has RT cores, so Isaac Sim is a SUPPORTED config on
# it (A100/H200 have none -- headless physics may work there, but rendering,
# cameras and video do not). 48 GB is ample for a single-GPU PhysX sim + a
# ~400k-param MLP.
# VERIFY ON ARC: partition / gres / QOS names --
#   sinfo -o "%P %G %D" | grep -i l40s
#   sacctmgr show qos format=name%32,priority,maxwall | grep -i l40s
# If a short/high-priority QOS exists for L40S, add it on the CLI (--qos=...).
# If the L40S nodes belong to a different ARC cluster than the A100s (Falcon vs
# Tinkercliffs), submit from THAT cluster's login node.
# Fallback (headless, no cameras/video, Tier 0 risk applies):
#   --partition=a100_normal_q --qos=tc_a100_normal_short --gres=gpu:a100:1
# =============================================================================
#SBATCH --job-name=rtn-train
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
# --account and --mail-user are deliberately NOT here: this repo is public, and
# #SBATCH lines cannot read env vars. Pass both on the command line.
# --mem is explicit on purpose: --mem=0 (whole node) can never backfill.
# --cpus-per-task=16: terrain generation + A* solvability run on the CPU at
# every reset; starving them makes resets, not physics, the bottleneck.

# shellcheck source=arc/arc_env.sh
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "[rtn] START train ${SLURM_JOB_ID} on $(hostname): $*"

"$PY" training/train.py --config configs/train.yaml --headless "$@"
