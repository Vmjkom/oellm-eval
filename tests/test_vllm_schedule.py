import json
import os
import shlex
import subprocess
import sys

import pytest

from oellm.main import schedule_evals


@pytest.fixture
def schedule(tmp_path, monkeypatch):
    monkeypatch.setattr("oellm.main._load_cluster_env", lambda: None)
    monkeypatch.setattr("oellm.main._num_jobs_in_queue", lambda: 0)
    for key, value in {
        "EVAL_OUTPUT_DIR": str(tmp_path / "output"),
        "EVAL_BASE_DIR": str(tmp_path),
        "EVAL_CONTAINER_IMAGE": "eval.sif",
        "HF_HOME": str(tmp_path / "hf"),
        "GPUS_PER_NODE": "1",
        "QUEUE_LIMIT": "1",
        "NODES": "1",
        "SINGULARITY_ARGS": "--rocm --contain",
    }.items():
        monkeypatch.setenv(key, value)

    def generate(**kwargs):
        opts = {
            "models": "test/model",
            "tasks": "test_task",
            "n_shot": 0,
            "skip_checks": True,
            "dry_run": True,
        }
        opts.update(kwargs)
        schedule_evals(**opts)
        script = next((tmp_path / "output").glob("*/submit_evals.sbatch"))
        subprocess.run(["bash", "-n", str(script)], check=True)
        return script

    return generate


def test_hf_default(schedule):
    script = schedule().read_text()
    assert 'MODEL_BACKEND="hf"' in script
    assert 'EVAL_BATCH_SIZE="auto"' in script


def test_tensor_parallel_allocation(schedule):
    script = schedule(model_backend="vllm", tensor_parallel_size=8).read_text()
    assert "#SBATCH --gres=gpu:8" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "tensor_parallel_size=8" in script


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"model_backend": "unknown"}, "model_backend"),
        ({"tensor_parallel_size": 2}, "requires"),
        ({"model_backend": "vllm", "tensor_parallel_size": 0}, "positive"),
        ({"batch_size": "0"}, "batch_size"),
        ({"model_args": "data_parallel_size=8"}, "not supported"),
        ({"model_backend": "vllm", "slurm_template_var": '{"NODES":2}'}, "one node"),
    ],
)
def test_invalid_settings(schedule, kwargs, match):
    with pytest.raises(ValueError, match=match):
        schedule(**kwargs)


def test_reject_mixed_suites(schedule, tmp_path):
    csv = tmp_path / "mixed.csv"
    csv.write_text(
        "model_path,task_path,n_shot,eval_suite\nm,t,0,lm_eval\nm,u,0,lighteval\n"
    )
    with pytest.raises(ValueError, match="lighteval"):
        schedule(
            models=None,
            tasks=None,
            n_shot=None,
            eval_csv_path=str(csv),
            model_backend="vllm",
        )


@pytest.mark.parametrize("container", [False, True])
def test_launch_arguments_masks_and_failure(schedule, tmp_path, monkeypatch, container):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").write_text(f"export PATH={shlex.quote(str(bin_dir))}:$PATH\n")
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\nimport os, sys, json\n"
        "with open(os.environ['CAPTURE_PATH'], 'a') as f:\n"
        "    f.write(json.dumps({'args': sys.argv[1:], 'masks': "
        "[os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', "
        "'ROCR_VISIBLE_DEVICES']]}) + '\\n')\n"
        "sys.exit(int(os.environ.get('FAKE_EXIT', '0')))\n"
    )
    python.chmod(0o755)
    singularity = bin_dir / "singularity"
    singularity.write_text(
        '#!/bin/bash\nwhile [[ "$1" != *.sif ]]; do shift; done\nshift\nexec "$@"\n'
    )
    singularity.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    capture = tmp_path / "calls.jsonl"
    monkeypatch.setenv("CAPTURE_PATH", str(capture))
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "1,3",
        "HIP_VISIBLE_DEVICES": "1,3",
        "ROCR_VISIBLE_DEVICES": "0,2,4,6",
        "WORLD_SIZE": "1",
        "LMEVAL_DP": "1",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ID": "123",
    }.items():
        monkeypatch.setenv(key, value)
    # Literal shell metacharacters in user arguments must never execute/expand.
    args = "revision=$HOME,tag=$(touch SHOULD_NOT_EXIST),note='quoted'"
    script = schedule(
        model_backend="vllm",
        tensor_parallel_size=2,
        batch_size="512",
        model_args=args,
        tasks="first,second",
        venv_path=None if container else str(bin_dir.parent),
    )
    monkeypatch.setenv("FAKE_EXIT", "7")
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 1, result.stdout + result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 2  # A failed task does not suppress subsequent tasks.
    for call in calls:
        argv = call["args"]
        assert argv[argv.index("--model") + 1] == "vllm"
        assert argv[argv.index("--batch_size") + 1] == "512"
        assert argv[argv.index("--model_args") + 1] == (
            f"pretrained=test/model,trust_remote_code=True,{args},tensor_parallel_size=2"
        )
        assert call["masks"] == ["1,3", "1,3", "0,2,4,6"]
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()

    capture.unlink()
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert subprocess.run(["bash", str(script)], capture_output=True).returncode == 1
    assert not capture.exists()
