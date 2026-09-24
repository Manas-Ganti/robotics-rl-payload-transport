#!/bin/bash
# =============================================================================
# record.slurm -- demo MP4s from a trained checkpoint (eval/record_clips.py).
# =============================================================================
# Visuals only; runs AFTER training. Needs the renderer, so an RT-core GPU
# (L40S) -- not the A100/H200 fallback. Submit FROM THE REPO ROOT; args pass
# straight to eval/record_clips.py.
#
#   sbatch --account=<ACCT> --mail-user=<you>@vt.edu arc/record.slurm \
#          --checkpoint results/runs/p1/model_final.pt --tag p1 \
#          --set robot.usd_path=/home/<you>/isaac_assets/carter/carter_v1.usd
#
# Output: results/clips/<tag>/*.mp4 + clips_manifest.json (params + outcome
# of every clip -- caption from it; success RATES only from the eval tables).
# =============================================================================
#SBATCH --job-name=rtn-record
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

# shellcheck source=arc/arc_env.sh
source "${SLURM_SUBMIT_DIR:-.}/arc/arc_env.sh"
trap arc_notify_finish EXIT
arc_notify "[rtn] START record ${SLURM_JOB_ID} on $(hostname): $*"

"$PY" eval/record_clips.py --headless "$@"
