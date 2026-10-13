#!/bin/bash
#SBATCH --account=project_462001516
#SBATCH -J oellm-eval-test
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=7    # More than 8 increases billing
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH -o /flash/project_462001516/users/villekom/eval-output/slurm_logs/%x-%j.log

set -xeu

CONTAINER=/flash/project_462001516/containers/staging/oellm_eval_lumi_06_10_2026.sif
REPO=/pfs/lustrep1/scratch/project_462001516/users/villekom/oellm-eval
model="openeurollm/datamix-flag-top_level-0.7b-option_5-100bt"

export VLLM_LOGGING_LEVEL=DEBUG
export TRITON_CACHE_DIR=/tmp/$USER/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/.cache/torchinductor
export SINGULARITY_BIND=/scratch/project_465002530,/flash/project_465002530,/flash/project_462001516,/scratch/project_462001516
export SINGULARITYENV_SLURM_MEM=60G
export EVAL_OUTPUT_DIR=/flash/project_462001516/users/villekom/eval-output
# Let Slurm select visible GPUs; one vLLM engine uses them for tensor parallelism.

srun singularity exec \
    --bind "$REPO:/opt/oellm-eval:ro" \
    "$CONTAINER" \
    /opt/venv/bin/oellm-eval schedule \
        --models "$model" \
        --tasks arc_easy,hellaswag,jeopardy \
        --n_shot 10 \
        --limit 500 \
        --model_backend vllm \
        --batch_size auto \
        --model_args dtype=bfloat16,max_length=4096,gpu_memory_utilization=0.9 \
        --local true \
        --venv_path /opt/venv \
        --skip_checks true

sacct -X -j $SLURM_JOBID --format=JobName%30,CPUTimeRAW%10,Elapsed%10,AveCPU%10,AveCPUFreq%10,AveDiskRead%10,ReqTRES%50
sacct --long -j $SLURM_JOBID -p > $EVAL_OUTPUT_DIR/sacct_output.csv

#if [ $SBATCH_PARTITION == "dev-g"]; then
#    USED-GPU-HOURS=
# LUMI compute billing if --cpus-per-task > 8. Look for more info here: https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/#partitions-allocatable-by-resources
#GPU-hours-billed = (
#    max(
#        ceil(CPU-cores-allocated / 8),
#        ceil(memory-allocated / 64GB),
#        GCDs-allocated )
#    * runtime-of-job) * 0.5
