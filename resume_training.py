#!/usr/bin/env python3
"""
Interactive script to continue MAPPO training for the SimCity Scale-Up Koto environment.

Features:
- Lists existing saved runs under results/models/ (those starting with mappo_seed)
- Shows most recent (max timestep) checkpoint per run
- Lets user choose which run and which specific step to resume
- Builds and executes the correct resume command (python main.py ... with checkpoint_path=... load_step=...)
- Supports non-interactive flags for automation.

Usage examples:
    python resume_training.py                # interactive
    python resume_training.py --run-index 0  # auto pick run 0 and latest step
    python resume_training.py --run-index 1 --step 50000 --t-max 10000000 --label resumed
    python resume_training.py --dry-run      # show command only

Note: Sacred CLI treats spaces as separators in parameter values. Our saved run directories contain spaces
(from the timestamp). We therefore escape spaces with '\\ ' for the checkpoint_path argument.
If this is fragile on your shell, consider renaming the directory or creating a symlink without spaces.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import subprocess
from typing import List, Optional, Tuple

RE_RUN_PREFIX = re.compile(r"^mappo_seed.*simcity_scale_up_koto_.*")
RE_STEP_DIR = re.compile(r"^[0-9]+$")

ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "results", "models")
SRC_DIR = os.path.join(ROOT, "src")


def find_runs() -> List[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    runs = []
    for name in sorted(os.listdir(MODELS_DIR)):
        full = os.path.join(MODELS_DIR, name)
        if os.path.isdir(full) and RE_RUN_PREFIX.match(name):
            runs.append(full)
    return runs


def find_steps(run_dir: str) -> List[int]:
    steps = []
    for name in os.listdir(run_dir):
        if RE_STEP_DIR.match(name):
            steps.append(int(name))
    steps.sort()
    return steps


def format_bytes(num: int) -> str:
    for unit in ['','K','M','G','T']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}B"
        num /= 1024.0
    return f"{num:.1f}PB"


def build_command(args, run_dir: str, step: int) -> List[str]:
    # Escape spaces for sacred CLI param token
    escaped_ckpt = run_dir.replace(' ', '\\ ')
    cmd = [
        sys.executable,
        "main.py",
        "--config=mappo",
        "--env-config=simcity_scale_up_koto",
        "with",
        f"checkpoint_path={escaped_ckpt}",
        f"load_step={step}",
        f"t_max={args.t_max}",
    ]
    if args.label:
        cmd.append(f"label={args.label}")
    if args.extra:
        # pass through extra raw sacred params
        for item in args.extra:
            cmd.append(item)
    return cmd


def interactive_select(runs: List[str]) -> int:
    print("Available runs:")
    for idx, run in enumerate(runs):
        steps = find_steps(run)
        latest = steps[-1] if steps else None
        print(f"[{idx}] {os.path.basename(run)} | checkpoints: {len(steps)} | latest: {latest}")
    while True:
        raw = input("Select run index (q to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Aborted.")
            sys.exit(0)
        if raw.isdigit() and 0 <= int(raw) < len(runs):
            return int(raw)
        print("Invalid selection.")


def interactive_step(steps: List[int]) -> int:
    latest = steps[-1]
    raw = input(f"Step to load (enter for latest {latest}): ").strip()
    if not raw:
        return latest
    if raw.isdigit() and int(raw) in steps:
        return int(raw)
    print("Invalid step, using latest.")
    return latest


def parse_args():
    p = argparse.ArgumentParser(description="Resume MAPPO training (SimCity Scale-Up Koto)")
    p.add_argument("--run-index", type=int, help="Index of run to resume (see list). If omitted, interactive.")
    p.add_argument("--step", type=int, help="Specific checkpoint step to load (default: latest).")
    p.add_argument("--t-max", type=int, default=10_000_000, help="New total t_max to train toward (default 10M).")
    p.add_argument("--label", type=str, help="Optional label for resumed run (sacred).")
    p.add_argument("--dry-run", action="store_true", help="Print command and exit without executing.")
    p.add_argument("--auto", action="store_true", help="Non-interactive: use provided --run-index and latest step if --step missing.")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Additional sacred params appended after main ones.")
    return p.parse_args()


def main():
    args = parse_args()

    runs = find_runs()
    if not runs:
        print("No runs found under results/models/. Start a training run first.")
        return 1

    if args.run_index is None:
        if args.auto:
            print("--auto given but no --run-index; cannot proceed.")
            return 1
        run_index = interactive_select(runs)
    else:
        if not (0 <= args.run_index < len(runs)):
            print(f"run-index {args.run_index} out of range (0..{len(runs)-1}).")
            return 1
        run_index = args.run_index

    run_dir = runs[run_index]
    steps = find_steps(run_dir)
    if not steps:
        print(f"No numeric checkpoint subdirectories found in {run_dir}")
        return 1

    if args.step is None:
        if args.auto:
            step = steps[-1]
        else:
            step = interactive_step(steps)
    else:
        if args.step not in steps:
            print(f"Requested step {args.step} not found. Available: {steps}")
            return 1
        step = args.step

    cmd = build_command(args, run_dir, step)

    print("\nResume configuration:")
    print(f"  Run directory : {run_dir}")
    print(f"  Checkpoint    : {step}")
    print(f"  t_max target  : {args.t_max}")
    if args.label:
        print(f"  Label         : {args.label}")
    print("  Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    if not args.auto:
        confirm = input("Proceed? [Y/n]: ").strip().lower()
        if confirm and confirm not in {"y", "yes"}:
            print("Aborted.")
            return 0

    # Execute from src directory so relative imports behave like original train script
    if not os.path.isdir(SRC_DIR):
        print(f"Cannot find src directory at {SRC_DIR}")
        return 1

    print("\nLaunching training...\n")
    try:
        result = subprocess.call(cmd, cwd=SRC_DIR)
        return result
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
