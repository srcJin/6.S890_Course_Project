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
from typing import List, Optional, Tuple, Dict

# Accept legacy (simcity_scale_up) and new (simcity_scale_up_koto) naming
RE_RUN_PREFIX = re.compile(r"^mappo_seed.*simcity_scale_up(_koto)?_.*")
RE_STEP_DIR = re.compile(r"^[0-9]+$")

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")

# Candidate model directories (training script cd's into src before running main.py)
MODEL_DIR_CANDIDATES = [
    os.path.join(ROOT, "results", "models"),            # root-level (older runs)
    os.path.join(SRC_DIR, "results", "models"),          # src-level (current runs)
]


def find_runs() -> List[Tuple[str, str]]:
    """Return list of (run_path, origin_tag). origin_tag indicates which models dir."""
    runs: List[Tuple[str, str]] = []
    for root_dir in MODEL_DIR_CANDIDATES:
        if not os.path.isdir(root_dir):
            continue
        try:
            for name in sorted(os.listdir(root_dir)):
                full = os.path.join(root_dir, name)
                if os.path.isdir(full) and RE_RUN_PREFIX.match(name):
                    runs.append((full, os.path.basename(root_dir)))
        except Exception:
            continue
    # Sort deterministically by path name
    runs.sort(key=lambda x: x[0])
    return runs


def detect_env_tag(run_path: str) -> str:
    base = os.path.basename(run_path)
    if "_koto_" in base:
        return "koto"
    return "base"


def build_run_metadata(runs: List[Tuple[str, str]]) -> List[Dict[str, object]]:
    meta = []
    for path, origin in runs:
        steps = find_steps(path)
        latest = steps[-1] if steps else None
        env_tag = detect_env_tag(path)
        meta.append({
            "path": path,
            "origin": origin,
            "env": env_tag,
            "n_ckpts": len(steps),
            "latest": latest,
        })
    return meta


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
    # No shell escaping needed because we pass an argv list; keep raw path.
    cmd = [
        sys.executable,
        "main.py",
        "--config=mappo",
        "--env-config=simcity_scale_up_koto",
        "with",
        f"checkpoint_path={run_dir}",
        f"load_step={step}",
        f"t_max={args.t_max}",
    ]
    if args.label:
        cmd.append(f"label={args.label}")
    # Auto reuse tensorboard logging unless disabled or user already specified
    if not args.no_reuse_tb:
        user_set = any(s.startswith("reuse_tb_logging=") for s in (args.extra or []))
        if not user_set:
            cmd.append("reuse_tb_logging=True")
    if args.extra:
        for item in args.extra:
            cmd.append(item)
    return cmd


EXPECTED_FILES = ["agent.th", "agent_opt.th", "critic.th", "critic_opt.th"]


def validate_checkpoint(run_dir: str, step: int) -> Optional[str]:
    # Handle previously escaped path stored in configs (remove backslash before space)
    cleaned = run_dir.replace("\\ ", " ")
    if cleaned != run_dir and os.path.isdir(cleaned):
        run_dir = cleaned
    step_dir = os.path.join(run_dir, str(step))
    if not os.path.isdir(step_dir):
        return f"Checkpoint directory missing: {step_dir}"
    missing = [f for f in EXPECTED_FILES if not os.path.isfile(os.path.join(step_dir, f))]
    if missing:
        return f"Checkpoint {step_dir} missing files: {', '.join(missing)}"
    return None


def interactive_select(runs: List[Tuple[str, str]], env_filter: Optional[str]) -> int:
    meta = build_run_metadata(runs)
    if env_filter:
        meta = [m for m in meta if m["env"] == env_filter]
        if not meta:
            print(f"No runs match env_filter='{env_filter}'.")
            sys.exit(1)
    print("Available runs:")
    print("Idx | Env  | Ckpts | Latest  | Origin   | Directory")
    print("----+------+-------+---------+----------+----------------------------------------------")
    for idx, m in enumerate(meta):
        print(f"{idx:>3} | {m['env']:<4} | {m['n_ckpts']:>5} | {str(m['latest']):>7} | {m['origin']:<8} | {os.path.basename(m['path'])}")
    while True:
        raw = input("Select run index (q to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Aborted.")
            sys.exit(0)
        if raw.isdigit():
            sel = int(raw)
            if 0 <= sel < len(meta):
                # Map back to original runs list index
                chosen_path = meta[sel]["path"]
                for i, (rp, _) in enumerate(runs):
                    if rp == chosen_path:
                        return i
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
    p.add_argument("--env-filter", choices=["koto", "base"], help="Filter listed runs by environment tag.")
    p.add_argument("--no-reuse-tb", action="store_true", help="Disable automatic tensorboard log reuse when resuming.")
    return p.parse_args()


def main():
    args = parse_args()

    runs = find_runs()
    if not runs:
        print("No runs found under any of: ")
        for d in MODEL_DIR_CANDIDATES:
            print("  -", d)
        print("Start a training run first.")
        return 1

    if args.run_index is None:
        if args.auto:
            print("--auto given but no --run-index; cannot proceed.")
            return 1
        run_index = interactive_select(runs, args.env_filter)
    else:
        if not (0 <= args.run_index < len(runs)):
            print(f"run-index {args.run_index} out of range (0..{len(runs)-1}).")
            return 1
        run_index = args.run_index

    run_dir = runs[run_index][0]
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

    # Pre-launch validation
    err = validate_checkpoint(run_dir, step)
    if err:
        print("Validation failed:", err)
        print("Available steps:", steps[-10:])
        return 1

    cmd = build_command(args, run_dir, step)

    print("\nResume configuration:")
    print(f"  Run directory : {run_dir}")
    print(f"  Checkpoint    : {step}")
    print(f"  t_max target  : {args.t_max}")
    if args.label:
        print(f"  Label         : {args.label}")
    print("  Command (argv list, spaces handled safely):")
    print(" ".join(cmd))
    print("  For manual shell copy you MAY need to escape spaces in checkpoint_path value.")

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
