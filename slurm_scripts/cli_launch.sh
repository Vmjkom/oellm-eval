SBATCH_CPUS_PER_GPU=7 QUEUE_LIMIT=2 .venv-launcher/bin/oellm-eval schedule \
  --models "openeurollm/datamix-flag-top_level-0.7b-option_5-100bt" \
  --tasks "arc_easy,hellaswag,jeopardy,winogrande" \
  --n_shot 10 \
  --model_backend vllm \
  --model_args "dtype=bfloat16,gpu_memory_utilization=0.9" \
  --limit 10 \
  --max_array_len 2 \
  --slurm_template_var "$(cat overrides.json)" \
  --dry_run True