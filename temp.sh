#!/bin/bash
#SBATCH --account=project_462001516
#SBATCH -J oellm-eval-test
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -c 7
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH -o /flash/project_462001516/users/villekom/eval-output/slurm_logs/%x-%j.log

set -xeu

CONTAINER=/flash/project_462001516/containers/staging/temp_sandbox
model="openeurollm/datamix-flag-top_level-0.7b-option_5-100bt"
export SINGULARITY_BIND=/scratch/project_465002530,/flash/project_465002530,/flash/project_462001516,/scratch/project_462001516
export SINGULARITYENV_SLURM_MEM=60G
export EVAL_OUTPUT_DIR=/flash/project_462001516/users/villekom/eval-output

# Maybe add this to singularity mounts /var/spool/slurmd

srun singularity exec \
    "$CONTAINER" \
        oellm-eval schedule \
            --models "$model" \
            --task_groups "dclm-core-22" \
            --n_shot 0 \
            --local true \
            --venv_path /opt/venv \
            --skip_checks true