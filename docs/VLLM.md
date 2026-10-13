# vLLM evaluations

Use the standard lm-eval-harness vLLM backend with `--model_backend vllm`.
Each evaluator call runs one task and saves its own result. There are no task
workers or combined task batches. Existing Slurm array scheduling still applies;
`--local true` runs tasks sequentially in the current allocation.

```bash
oellm-eval schedule \
  --models openeurollm/datamix-flag-top_level-0.7b-option_5-100bt \
  --task_groups dclm-core-22 \
  --model_backend vllm \
  --tensor_parallel_size 1 \
  --batch_size 512 \
  --model_args dtype=bfloat16,gpu_memory_utilization=0.9 \
  --local true --venv_path /opt/venv --skip_checks true
```

The runtime environment must contain lm-eval-harness and a compatible vLLM
installation. On LUMI, use the ROCm vLLM container; do not replace its vLLM with
a CUDA wheel. The CLI itself can run from a separate environment when scheduling
container jobs. With `--local`, `--venv_path` identifies the evaluation environment
(including when the CLI is already inside a container).

`--tensor_parallel_size N` requests N GPUs and one node per generated Slurm job.
For local execution, allocate those GPUs yourself. One launcher starts the vLLM
engine, which manages tensor parallelism; do not wrap it in a distributed launcher.
CUDA/HIP/ROCR visibility masks are preserved. Backend options currently require
lm-eval-harness tasks, not Lighteval or Evalchemy.

`--batch_size` defaults to `auto`, preserving the existing harness behavior.
For large multiple-choice scoring tasks, a finite value such as 512 bounds the
number of prompt-logprob outputs accumulated per engine call. This is separate
from vLLM's GPU scheduler batch size and does not cap all host memory usage.
512 is a starting point, not a measured optimum.

The repository's `temp.sh` uses one GPU by default and runs tasks sequentially:

```bash
sbatch temp.sh
# Use eight GPUs for tensor parallelism within each task:
sbatch --gpus-per-node=8 temp.sh
```

The script passes the allocated GPU count as tensor-parallel size, uses a finite
batch size of 512, and retains its existing 60G memory and 90-minute time limits.
The earlier DCLM runs did not complete within 90 minutes; this implementation
does not promise that the full suite fits that limit. Small models may gain
little from tensor parallelism. HF remains the default backend.
