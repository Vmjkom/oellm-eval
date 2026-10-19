#!/bin/bash
#SBATCH --account=project_462001516
#SBATCH -J oellm-eval-test
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=7    # More than 8 increases billing
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH -o /flash/project_462001516/users/villekom/eval-output/slurm_logs/%x-%j.log

set -xeu

CONTAINER=/flash/project_462001516/containers/staging/oellm_eval_lumi_06_10_2026.sif
REPO=/pfs/lustrep1/scratch/project_462001516/users/villekom/oellm-eval
model="openeurollm/datamix-flag-top_level-0.7b-option_5-100bt"

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CACHE_ROOT=/tmp/$USER/.cache/vllm/torch_compile
export TRITON_CACHE_DIR=/tmp/$USER/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/.cache/torchinductor
export SINGULARITY_BIND=/scratch/project_465002530,/flash/project_465002530,/flash/project_462001516,/scratch/project_462001516
export SINGULARITYENV_SLURM_MEM=60G
export EVAL_OUTPUT_DIR=/flash/project_462001516/users/villekom/eval-output

srun singularity exec \
    --bind "$REPO:/opt/oellm-eval:ro" \
    "$CONTAINER" \
    /opt/venv/bin/oellm-eval schedule \
        --models "$model" \
        --tasks arc_easy,hellaswag,jeopardy \
        --n_shot 10 \
        --model_backend hf \
        --batch_size auto \
        --model_args dtype=bfloat16,max_length=4096 \
        --local true \
        --venv_path /opt/venv \
        --skip_checks true

sacct -X -j $SLURM_JOBID --format=JobName%30,CPUTimeRAW%10,Elapsed%10,AveCPU%10,AveCPUFreq%10,AveDiskRead%10,ReqTRES%50
sacct --long -j "$SLURM_JOB_ID" -p > "$EVAL_OUTPUT_DIR/sacct-${SLURM_JOB_ID}-hf.csv"
