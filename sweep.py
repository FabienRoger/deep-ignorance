#!/usr/bin/env python3
"""Launch hyperparameter sweep across multiple GPUs.

Sweeps over learning rates with 3 training modes:
- Pure NTP (no teacher)
- Pure KD (kd_alpha=1.0, no hidden supervision)
- KD+MSE (kd_alpha=0.5, hidden supervision)

Each GPU runs one job at a time.
"""

import subprocess
import time
from itertools import cycle
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these variables
# ============================================================================

# Data and model paths
DATA_PATH = "filtered_output_test/retained_dataset.jsonl"
STUDENT_MODEL = "EleutherAI/deep-ignorance-random-init"
TEACHER_MODEL = "EleutherAI/deep-ignorance-unfiltered"

# Training settings
# NUM_STEPS = 2000
NUM_STEPS = 20000
BATCH_SIZE = 2
SAVE_INTERVAL = 0
EVAL_EVERY = 1000

# Sweep settings
# GPUS = [4, 5, 6, 7]
GPUS = [5, 6, 7]
# LEARNING_RATES = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
# LEARNING_RATES = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
LEARNING_RATES = [3e-5]
MODES = ["ntp", "kd", "kd_mse"]  # ntp, kd, kd_mse

# Logging
WANDB_PROJECT = "deep-ignorance-sweep"
POLL_INTERVAL = 10  # Seconds between checking processes

# ============================================================================


def run_experiment(gpu_id, lr, mode):
    """Run a single experiment on a specific GPU.

    Args:
        gpu_id: GPU device ID
        lr: Learning rate
        mode: Training mode ('ntp', 'kd', or 'kd_mse')
    """
    # Create run name
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    run_name = f"v5l_{mode}_lr{lr_str}_bs{BATCH_SIZE}"

    # Build command
    cmd = [
        "python",
        "finetune_simple.py",
        "--data_path",
        DATA_PATH,
        "--student_model",
        STUDENT_MODEL,
        "--output_dir",
        f"./checkpoints/sweep/{run_name}",
        "--num_steps",
        str(NUM_STEPS),
        "--batch_size",
        str(BATCH_SIZE),
        "--lr",
        str(lr),
        "--use_bf16",
        "--use_wandb",
        "--wandb_project",
        WANDB_PROJECT,
        "--wandb_run_name",
        run_name,
        "--save_interval",
        str(SAVE_INTERVAL),
        "--eval_every",
        str(EVAL_EVERY),
    ]

    # Add mode-specific arguments
    if mode in ["kd", "kd_mse"]:
        cmd.extend(
            [
                "--teacher_model",
                TEACHER_MODEL,
            ]
        )

        if mode == "kd":
            # Pure KD: kd_alpha=1.0, no hidden supervision
            cmd.extend(
                [
                    "--kd_alpha",
                    "1.0",
                ]
            )
        elif mode == "kd_mse":
            cmd.extend(
                [
                    "--kd_alpha",
                    "1.0",
                    "--hidden_supervision",
                    "--hidden_loss_weight",
                    "1.0",
                ]
            )

    # Set environment variable for GPU
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    print(f"[GPU {gpu_id}] Starting: {run_name}")
    print(f"[GPU {gpu_id}] Command: {' '.join(cmd)}")

    # Run the command (don't capture output to avoid pipe buffer issues)
    process = subprocess.Popen(
        cmd,
        env={**subprocess.os.environ.copy(), **env},
    )

    return process, run_name


def main():
    print(f"Learning rates to sweep: {LEARNING_RATES}")
    print(f"Modes to sweep: {MODES}")
    print(f"Total experiments: {len(LEARNING_RATES) * len(MODES)}")
    print(f"GPUs: {GPUS}")
    print()

    # Create output directory
    Path("./checkpoints/sweep").mkdir(parents=True, exist_ok=True)

    # Create job queue: (lr, mode) pairs
    jobs = [(lr, mode) for mode in MODES for lr in LEARNING_RATES]
    # shuffle the jobs
    import random

    random.Random(42).shuffle(jobs)

    # Track running processes: {gpu_id: (process, run_name)}
    running = {gpu_id: None for gpu_id in GPUS}

    # Cycle through GPUs
    gpu_cycle = cycle(GPUS)
    job_idx = 0

    print("Starting sweep...")
    print("=" * 80)

    try:
        while job_idx < len(jobs) or any(proc is not None for proc in running.values()):
            # Check for finished processes and start new ones
            for gpu_id in GPUS:
                if running[gpu_id] is not None:
                    process, run_name = running[gpu_id]
                    retcode = process.poll()

                    if retcode is not None:
                        # Process finished
                        if retcode == 0:
                            print(f"[GPU {gpu_id}] ✓ Completed: {run_name}")
                        else:
                            print(f"[GPU {gpu_id}] ✗ Failed (code {retcode}): {run_name}")

                        running[gpu_id] = None

                # Start new job if GPU is free and jobs remain
                if running[gpu_id] is None and job_idx < len(jobs):
                    lr, mode = jobs[job_idx]
                    process, run_name = run_experiment(gpu_id, lr, mode)
                    running[gpu_id] = (process, run_name)
                    job_idx += 1
                    print()

            # Print status
            active_jobs = sum(1 for proc in running.values() if proc is not None)
            print(
                f"[Status] Active: {active_jobs}/{len(GPUS)} GPUs | "
                f"Completed: {job_idx - active_jobs}/{len(jobs)} | "
                f"Remaining: {len(jobs) - job_idx}"
            )

            # Wait before next check
            time.sleep(POLL_INTERVAL)

        print()
        print("=" * 80)
        print("Sweep complete!")

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("KeyboardInterrupt detected! Killing all running processes...")
        print("=" * 80)

        # Kill all running processes
        for gpu_id in GPUS:
            if running[gpu_id] is not None:
                process, run_name = running[gpu_id]
                print(f"[GPU {gpu_id}] Terminating: {run_name}")
                process.terminate()
                try:
                    # Wait up to 5 seconds for graceful termination
                    process.wait(timeout=5)
                    print(f"[GPU {gpu_id}] Terminated: {run_name}")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    print(f"[GPU {gpu_id}] Force killing: {run_name}")
                    process.kill()
                    process.wait()
                    print(f"[GPU {gpu_id}] Killed: {run_name}")

        print("=" * 80)
        print("All processes terminated. Exiting.")
        print("=" * 80)


if __name__ == "__main__":
    main()
