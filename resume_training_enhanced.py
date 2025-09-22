#!/usr/bin/env python3
"""
Enhanced resume training script for MAPPO training with full config inheritance.

Features:
- Automatically inherits ALL original training settings from Sacred config files
- Preserves original seed, learning rate, batch sizes, and all hyperparameters
- Supports both interactive and automated modes
- Validates checkpoint integrity before resuming
- Builds commands with complete parameter inheritance

Usage examples:
    python resume_training_enhanced.py                    # interactive mode
    python resume_training_enhanced.py --run-index 0     # auto pick run 0 and latest step
    python resume_training_enhanced.py --run-index 1 --step 50000 --t-max 200000000
    python resume_training_enhanced.py --dry-run         # show command only
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import subprocess
from typing import List, Optional, Tuple, Dict, Any

# Accept legacy (simcity_scale_up) and new (simcity_scale_up_koto) naming
RE_RUN_PREFIX = re.compile(r"^mappo_seed.*simcity_scale_up(_koto)?_.*")
RE_STEP_DIR = re.compile(r"^[0-9]+$")

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")

# Candidate model directories (training script cd's into src before running main.py)
MODEL_DIR_CANDIDATES = [
    os.path.join(ROOT, "results", "models"),  # root-level (older runs)
    os.path.join(SRC_DIR, "results", "models"),  # src-level (current runs)
]

# Sacred config directories
SACRED_DIR_CANDIDATES = [
    os.path.join(ROOT, "results", "sacred"),  # root-level
    os.path.join(SRC_DIR, "results", "sacred"),  # src-level
]

# Parameters that should NOT be inherited (we want to override these)
OVERRIDE_PARAMS = {
    "checkpoint_path",  # We set this to resume from checkpoint
    "load_step",  # We set this to the specific step
    "t_max",  # User may want to extend training
    "label",  # User may want custom label for resumed run
    "evaluate",  # Should be False for training
    "render",  # Should be False for training
    "reuse_tb_logging",  # We may want to control this separately
}


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
    runs.sort(key=lambda x: x[0])
    return runs


def detect_env_tag(run_path: str) -> str:
    """Detect environment type from run path."""
    base = os.path.basename(run_path)
    if "_koto_" in base:
        return "koto"
    return "base"


def find_steps(run_dir: str) -> List[int]:
    """Find available checkpoint steps in run directory."""
    steps = []
    try:
        for name in os.listdir(run_dir):
            if RE_STEP_DIR.match(name):
                steps.append(int(name))
    except Exception:
        pass
    steps.sort()
    return steps


def find_sacred_config(run_path: str) -> Optional[str]:
    """Find corresponding Sacred config file for a run."""
    # Extract identifying information from run path
    run_name = os.path.basename(run_path)

    # Try to extract seed from run name (e.g., mappo_seed996602412_simcity_scale_up_koto_...)
    seed_match = re.search(r"seed(\d+)", run_name)
    if not seed_match:
        return None

    seed = seed_match.group(1)

    # Determine environment name
    if "_koto_" in run_name:
        env_name = "simcity_scale_up_koto"
    else:
        env_name = "simcity_scale_up"

    # Search in Sacred directories
    for sacred_root in SACRED_DIR_CANDIDATES:
        if not os.path.exists(sacred_root):
            continue

        sacred_path = os.path.join(sacred_root, "mappo", env_name)
        if not os.path.exists(sacred_path):
            continue

        # Look through experiment directories
        try:
            for exp_dir in os.listdir(sacred_path):
                exp_path = os.path.join(sacred_path, exp_dir)
                if not os.path.isdir(exp_path):
                    continue

                config_path = os.path.join(exp_path, "config.json")
                if os.path.exists(config_path):
                    # Check if this config matches our seed
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        if str(config.get("seed", "")) == seed:
                            return config_path
                    except Exception:
                        continue
        except Exception:
            continue

    return None


def load_sacred_config(config_path: str) -> Dict[str, Any]:
    """Load and parse Sacred config file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}


def build_config_params(config: Dict[str, Any], overrides: Dict[str, Any]) -> List[str]:
    """Build parameter list from Sacred config, applying overrides."""
    params = []

    # Handle nested env_args
    env_args = config.get("env_args", {})
    for key, value in env_args.items():
        if key != "seed":  # Skip env seed, use global seed
            params.append(f"env_args.{key}={value}")

    # Handle main config parameters
    for key, value in config.items():
        if key in OVERRIDE_PARAMS:
            continue  # Skip parameters we want to override
        if key == "env_args":
            continue  # Already handled above
        if key in overrides:
            continue  # Will be set by overrides

        # Convert boolean values to proper format
        if isinstance(value, bool):
            params.append(f"{key}={str(value)}")
        else:
            params.append(f"{key}={value}")

    # Apply overrides
    for key, value in overrides.items():
        if isinstance(value, bool):
            params.append(f"{key}={str(value)}")
        else:
            params.append(f"{key}={value}")

    return params


def build_command_with_config(
    args, run_dir: str, step: int, config: Dict[str, Any]
) -> List[str]:
    """Build command with full config inheritance."""
    # Determine config files based on environment
    env_name = config.get("env", "simcity_scale_up_koto")
    if env_name == "simcity_scale_up_koto":
        env_config = "simcity_scale_up_koto"
    else:
        env_config = "simcity_scale_up"

    # Base command
    cmd = [
        sys.executable,
        "main.py",
        "--config=mappo",
        f"--env-config={env_config}",
        "with",
    ]

    # Prepare overrides
    overrides = {
        "checkpoint_path": run_dir,
        "load_step": step,
        "t_max": args.t_max,
        "evaluate": False,
        "render": False,
    }

    if args.label:
        overrides["label"] = args.label

    # Auto reuse tensorboard logging unless disabled
    if not args.no_reuse_tb:
        overrides["reuse_tb_logging"] = True

    # Build parameters from config
    config_params = build_config_params(config, overrides)
    cmd.extend(config_params)

    # Add any extra parameters
    if args.extra:
        cmd.extend(args.extra)

    return cmd


def build_run_metadata(runs: List[Tuple[str, str]]) -> List[Dict[str, object]]:
    """Build metadata for runs including Sacred config info."""
    meta = []
    for path, origin in runs:
        steps = find_steps(path)
        latest = steps[-1] if steps else None
        env_tag = detect_env_tag(path)

        # Try to find Sacred config
        sacred_config_path = find_sacred_config(path)
        has_config = sacred_config_path is not None

        # If we have config, try to extract seed
        seed = None
        if has_config:
            config = load_sacred_config(sacred_config_path)
            seed = config.get("seed")

        meta.append(
            {
                "path": path,
                "origin": origin,
                "env": env_tag,
                "n_ckpts": len(steps),
                "latest": latest,
                "has_config": has_config,
                "seed": seed,
                "config_path": sacred_config_path,
            }
        )
    return meta


EXPECTED_FILES = ["agent.th", "agent_opt.th", "critic.th", "critic_opt.th"]


def validate_checkpoint(run_dir: str, step: int) -> Optional[str]:
    """Validate checkpoint integrity."""
    cleaned = run_dir.replace("\\ ", " ")
    if cleaned != run_dir and os.path.isdir(cleaned):
        run_dir = cleaned
    step_dir = os.path.join(run_dir, str(step))
    if not os.path.isdir(step_dir):
        return f"Checkpoint directory missing: {step_dir}"
    missing = [
        f for f in EXPECTED_FILES if not os.path.isfile(os.path.join(step_dir, f))
    ]
    if missing:
        return f"Checkpoint {step_dir} missing files: {', '.join(missing)}"
    return None


def interactive_select(runs: List[Tuple[str, str]], env_filter: Optional[str]) -> int:
    """Interactive run selection."""
    meta = build_run_metadata(runs)
    if env_filter:
        meta = [m for m in meta if m["env"] == env_filter]
        if not meta:
            print(f"No runs match env_filter='{env_filter}'.")
            sys.exit(1)

    print("Available runs:")
    print("Idx | Env  | Ckpts | Latest  | Seed       | Config | Origin   | Directory")
    print(
        "----+------+-------+---------+------------+--------+----------+----------------------------------------------"
    )

    for idx, m in enumerate(meta):
        config_status = "✓" if m["has_config"] else "✗"
        seed_str = str(m["seed"]) if m["seed"] else "unknown"
        print(
            f"{idx:>3} | {m['env']:<4} | {m['n_ckpts']:>5} | {str(m['latest']):>7} | {seed_str:<10} | {config_status:>6} | {m['origin']:<8} | {os.path.basename(m['path'])}"
        )

    while True:
        raw = input("Select run index (q to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Aborted.")
            sys.exit(0)
        if raw.isdigit():
            sel = int(raw)
            if 0 <= sel < len(meta):
                chosen_path = meta[sel]["path"]
                for i, (rp, _) in enumerate(runs):
                    if rp == chosen_path:
                        return i
        print("Invalid selection.")


def interactive_step(steps: List[int]) -> int:
    """Interactive step selection."""
    latest = steps[-1]
    raw = input(f"Step to load (enter for latest {latest}): ").strip()
    if not raw:
        return latest
    if raw.isdigit() and int(raw) in steps:
        return int(raw)
    print("Invalid step, using latest.")
    return latest


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Enhanced resume MAPPO training with full config inheritance"
    )
    p.add_argument(
        "--run-index",
        type=int,
        help="Index of run to resume (see list). If omitted, interactive.",
    )
    p.add_argument(
        "--step", type=int, help="Specific checkpoint step to load (default: latest)."
    )
    p.add_argument(
        "--t-max",
        type=int,
        default=100_000_000,
        help="New total t_max to train toward (default 100M).",
    )
    p.add_argument("--label", type=str, help="Optional label for resumed run (sacred).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and exit without executing.",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive: use provided --run-index and latest step if --step missing.",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional sacred params appended after main ones.",
    )
    p.add_argument(
        "--env-filter",
        choices=["koto", "base"],
        help="Filter listed runs by environment tag.",
    )
    p.add_argument(
        "--no-reuse-tb",
        action="store_true",
        help="Disable automatic tensorboard log reuse when resuming.",
    )
    return p.parse_args()


def main():
    """Main function."""
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

    # Find and load Sacred config
    sacred_config_path = find_sacred_config(run_dir)
    if not sacred_config_path:
        print(f"Warning: No Sacred config found for run {run_dir}")
        print("Falling back to basic resume without full config inheritance.")
        # Fallback to original simple command
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
    else:
        print(f"Loading full config from: {sacred_config_path}")
        config = load_sacred_config(sacred_config_path)
        cmd = build_command_with_config(args, run_dir, step, config)

    print("\nResume configuration:")
    print(f"  Run directory : {run_dir}")
    print(f"  Checkpoint    : {step}")
    print(f"  t_max target  : {args.t_max}")
    print(f"  Sacred config : {sacred_config_path or 'Not found'}")
    if sacred_config_path:
        config = load_sacred_config(sacred_config_path)
        print(f"  Original seed : {config.get('seed', 'unknown')}")
        print(f"  Learning rate : {config.get('lr', 'unknown')}")
        print(f"  Batch size    : {config.get('batch_size', 'unknown')}")
    if args.label:
        print(f"  Label         : {args.label}")
    print("\n  Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    if not args.auto:
        confirm = input("Proceed? [Y/n]: ").strip().lower()
        if confirm and confirm not in {"y", "yes"}:
            print("Aborted.")
            return 0

    # Execute from src directory
    if not os.path.isdir(SRC_DIR):
        print(f"Cannot find src directory at {SRC_DIR}")
        return 1

    print("\nLaunching training with full config inheritance...\n")
    try:
        result = subprocess.call(cmd, cwd=SRC_DIR)
        return result
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
