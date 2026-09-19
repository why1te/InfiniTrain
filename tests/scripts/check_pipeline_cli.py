#!/usr/bin/env python3
"""Check both real example CLIs without GPUs, model assets, or parameter allocation.

Usage: python3 tests/scripts/check_pipeline_cli.py --build-dir /path/to/cpu-build
"""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', required=True, type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = os.environ.copy()
    env.update(WORLD_SIZE='1', LOCAL_WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
               OMP_NUM_THREADS='1', GLOG_logtostderr='1')
    cases = [
        (['--pipeline_layer_partition=1,'], 'is empty'),
        (['--pipeline_layer_partition=1', '--pipeline_chunk_layout=0:1'], 'mutually exclusive'),
        (['--pipeline_parallel=0'], 'PP/vPP must be positive'),
        (['--virtual_pipeline_parallel=2147483648'], 'PP/vPP must be positive'),
        (['--pipeline_layer_partition=1,1'], 'PP * vPP entries'),
        (['--pipeline_parallel=2', '--pipeline_chunk_layout=0:6,2:6'], 'invalid owner/count'),
        (['--pipeline_parallel=2', '--pipeline_chunk_layout=0:1,1:1', '--dtype=bfloat16',
          '--save=unused', '--save_interval=0', '--freq_generate_txt=0',
          '--val_loss_every=0', '--sample_every=0'], 'but --save_interval is 0'),
        # Valid syntax reaches the scratch model's actual size check, before model allocation.
        (['--pipeline_layer_partition=1', '--llmc_filepath=', '--device=cpu'], 'expected_sum='),
    ]
    with tempfile.TemporaryDirectory(prefix='pipeline-cli-') as working:
        for model in ('gpt2', 'llama3'):
            executable = (args.build_dir / model).resolve()
            if not executable.is_file():
                raise FileNotFoundError(executable)
            for flags, diagnostic in cases:
                case_env = env.copy()
                if '--dtype=bfloat16' in flags:
                    # Reach Train's existing configuration validation before CUDA setup.
                    case_env.update(WORLD_SIZE='2', LOCAL_WORLD_SIZE='2', RANK='1', LOCAL_RANK='1')
                result = subprocess.run([str(executable), *flags], cwd=working, env=case_env,
                                        capture_output=True, text=True, timeout=20)
                output = result.stdout + result.stderr
                if result.returncode == 0 or diagnostic not in output:
                    raise AssertionError(f'{model} {flags}: code={result.returncode}\n{output}')
    print(f'PASS: {len(cases) * 2} CLI checks; no GPU training performed')


if __name__ == '__main__':
    main()
