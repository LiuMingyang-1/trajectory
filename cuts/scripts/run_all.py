#!/usr/bin/env python3
"""Run the cuts experiment stages in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    script_path: Path


STAGES = [
    Stage(key="entropy", label="Entropy Extraction", script_path=SCRIPT_DIR / "run_entropy_extraction.py"),
    Stage(key="baseline", label="Baseline", script_path=SCRIPT_DIR / "run_baseline.py"),
    Stage(key="cut_a", label="Cut A", script_path=SCRIPT_DIR / "run_cut_a.py"),
    Stage(key="cut_b", label="Cut B", script_path=SCRIPT_DIR / "run_cut_b.py"),
    Stage(key="cut_c", label="Cut C", script_path=SCRIPT_DIR / "run_cut_c.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip_entropy", action="store_true", help="Skip entropy extraction.")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip the baseline pipeline.")
    parser.add_argument("--skip_cut_a", action="store_true", help="Skip Cut A.")
    parser.add_argument("--skip_cut_b", action="store_true", help="Skip Cut B.")
    parser.add_argument("--skip_cut_c", action="store_true", help="Skip Cut C.")
    parser.add_argument("--device", type=str, default="cpu", help="Device passed to stage scripts when supported.")
    parser.add_argument("--max_samples", type=int, default=None, help="Sample cap for entropy and baseline stages.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Optional model path passed to entropy extraction.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter used to invoke scripts.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for entropy extraction inference.")
    return parser.parse_args()


def should_skip(stage: Stage, args: argparse.Namespace) -> bool:
    return bool(getattr(args, f"skip_{stage.key}"))


def build_stage_cmd(stage: Stage, args: argparse.Namespace) -> list[str]:
    cmd = [args.python, str(stage.script_path)]

    if stage.key == "entropy":
        cmd.extend(["--device", args.device])
        if args.model_name_or_path is not None:
            cmd.extend(["--model_name_or_path", args.model_name_or_path])
        if args.max_samples is not None:
            cmd.extend(["--max_samples", str(args.max_samples)])
        if args.batch_size is not None:
            cmd.extend(["--batch_size", str(args.batch_size)])
        return cmd

    if stage.key == "baseline":
        cmd.extend(["--device", args.device, "--python", args.python])
        if args.max_samples is not None:
            cmd.extend(["--max_samples", str(args.max_samples)])
        return cmd

    if stage.key in {"cut_a", "cut_b", "cut_c"}:
        cmd.extend(["--device", args.device])
        return cmd

    raise ValueError(f"Unknown stage key: {stage.key}")


def format_cmd(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def print_summary(summary_rows: list[dict[str, str]]) -> None:
    print("\nFinal summary")
    for row in summary_rows:
        detail = row["detail"]
        if detail:
            print(f"- {row['label']}: {row['status']} ({detail})")
        else:
            print(f"- {row['label']}: {row['status']}")


def main() -> None:
    args = parse_args()

    print(f"Cuts root: {CUTS_ROOT}")
    print(f"Python: {args.python}")
    print(f"Device: {args.device}")
    if args.model_name_or_path is not None:
        print(f"Model: {args.model_name_or_path}")
    if args.max_samples is not None:
        print(f"Max samples: {args.max_samples}")

    summary_rows: list[dict[str, str]] = []
    failed = False
    exit_code = 0

    for index, stage in enumerate(STAGES, start=1):
        if should_skip(stage, args):
            print(f"\n[{index}/{len(STAGES)}] {stage.label}")
            print("Status: skipped by flag")
            summary_rows.append({"label": stage.label, "status": "skipped", "detail": "flag"})
            continue

        if not stage.script_path.exists():
            print(f"\n[{index}/{len(STAGES)}] {stage.label}")
            print(f"Status: skipped, script not found at {stage.script_path}")
            summary_rows.append({"label": stage.label, "status": "skipped", "detail": "missing script"})
            continue

        cmd = build_stage_cmd(stage, args)
        print(f"\n[{index}/{len(STAGES)}] {stage.label}")
        print(f"Script: {stage.script_path}")
        print(f"CMD: {format_cmd(cmd)}")
        result = subprocess.run(cmd, cwd=CUTS_ROOT)
        if result.returncode != 0:
            print(f"ERROR: {stage.label} failed with return code {result.returncode}")
            summary_rows.append(
                {
                    "label": stage.label,
                    "status": "failed",
                    "detail": f"return code {result.returncode}",
                }
            )
            failed = True
            exit_code = int(result.returncode)
            break

        print(f"Status: completed")
        summary_rows.append({"label": stage.label, "status": "ran", "detail": format_cmd(cmd)})

    if failed:
        remaining = STAGES[len(summary_rows) :]
        for stage in remaining:
            summary_rows.append({"label": stage.label, "status": "not run", "detail": "blocked by prior failure"})

    print_summary(summary_rows)
    if failed:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
