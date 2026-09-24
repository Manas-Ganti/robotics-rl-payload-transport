#!/bin/bash
# =============================================================================
# eval.slurm -- OOD grid sweep for an RL checkpoint, Nav2, or the random floor.
# =============================================================================
# Submit FROM THE REPO ROOT; args pass straight to eval/run_eval.py.
#
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu arc/eval.slurm \
#          --checkpoint results/runs/p2_slope_s42/model_final.pt --tag p2_s42
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu arc/eval.slurm --policy nav2   --tag nav2
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu arc/eval.slurm --policy random --tag random
#
# Always pass --tag: output files are named by it, and two sweeps written to the
# same results/eval/ without one are hard to tell apart later.
# =============================================================================
#SBATCH --job-name=rtn-eval
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
# VERIFY ON ARC: partition/gres/QOS names -- see arc/train.slurm header.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# shellcheck source=arc/arc_env.sh
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "[rtn] START eval ${SLURM_JOB_ID} on $(hostname): $*"

"$PY" eval/run_eval.py --config configs/eval_ood.yaml --headless "$@"
