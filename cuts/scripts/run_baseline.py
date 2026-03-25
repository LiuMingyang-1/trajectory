#!/usr/bin/env python3
"""Run the ICR-only baseline pipeline for cuts comparison experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPTS_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import (
    BASELINE_RESULTS_DIR,
    ICR_DATASET_DIR,
    ICR_INPUT_JSONL,
    ICR_INTERMEDIATE_DIR,
    ICR_REPO_ROOT,
    ICR_RESULTS_DIR,
    ICR_SPAN_CANDIDATE_DIR,
    ICR_SPAN_LABEL_DIR,
    QA_DATA_JSON,
)


ICR_SCRIPTS_DIR = ICR_REPO_ROOT / "scripts"
SPAN_READY_PATH = ICR_INTERMEDIATE_DIR / "icr_halu_eval_span_ready.jsonl"
WINDOWS_PATH = ICR_SPAN_CANDIDATE_DIR / "tokenizer_windows.jsonl"
SILVER_PATH = ICR_SPAN_LABEL_DIR / "tokenizer_windows_silver_labels.jsonl"
DATASET_PATH = ICR_DATASET_DIR / "tokenizer_windows_dataset.jsonl"
RUN_METADATA_DIR = BASELINE_RESULTS_DIR / "_run_metadata"
SILVER_POSITIVE_THRESHOLD = 0.65
SILVER_NEGATIVE_THRESHOLD = 0.75
DATASET_POOLING = "mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ICR-only baseline pipeline.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for baseline MLP training.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit rows for quick pipeline tests.")
    parser.add_argument("--window_sizes", type=str, default="3", help="Comma-separated tokenizer window sizes.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter for upstream scripts.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip steps when the expected output file or metrics already exist.",
    )
    parser.add_argument("--n_splits", type=int, default=None, help="Override cross-validation fold count.")
    parser.add_argument("--epochs", type=int, default=None, help="Override baseline MLP epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override baseline MLP batch size.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override baseline MLP learning rate.",
    )
    parser.add_argument("--patience", type=int, default=None, help="Override baseline MLP early stopping patience.")
    parser.add_argument("--seed", type=int, default=None, help="Override shared training seed.")
    parser.add_argument(
        "--rf_estimators",
        type=int,
        default=None,
        help="Override discrepancy random forest estimator count.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Override discrepancy logistic regression max iterations.",
    )
    return parser.parse_args()


def verify_required_inputs() -> None:
    if not ICR_REPO_ROOT.exists():
        print(f"ERROR: ICR repo not found at {ICR_REPO_ROOT}")
        sys.exit(1)
    if not ICR_INPUT_JSONL.exists():
        print(f"ERROR: ICR input not found at {ICR_INPUT_JSONL}")
        sys.exit(1)
    if not QA_DATA_JSON.exists():
        print(f"ERROR: QA data not found at {QA_DATA_JSON}")
        sys.exit(1)


def run_step(
    cmd: list[str],
    step_name: str,
    *,
    skip: bool = False,
    skip_path: Path | None = None,
    skip_detail: str | None = None,
) -> bool:
    if skip:
        print(f"\n[SKIP] {step_name}")
        if skip_path is not None:
            print(f"Existing output: {skip_path}")
        if skip_detail:
            print(f"Reason: {skip_detail}")
        return False

    print(f"\n[STEP] {step_name}")
    print(f"CMD: {' '.join(str(part) for part in cmd)}")
    result = subprocess.run(cmd, cwd=ICR_REPO_ROOT)
    if result.returncode != 0:
        print(f"ERROR: {step_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[DONE] {step_name}")
    return True


def build_prepare_span_ready_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        str(ICR_SCRIPTS_DIR / "prepare_span_ready_data.py"),
        "--input_path",
        str(ICR_INPUT_JSONL),
        "--qa_data_path",
        str(QA_DATA_JSON),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    return cmd


def build_tokenizer_windows_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        str(ICR_SCRIPTS_DIR / "build_tokenizer_windows.py"),
        "--input_path",
        str(SPAN_READY_PATH),
        "--window_sizes",
        args.window_sizes,
    ]
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    return cmd


def build_silver_labels_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(ICR_SCRIPTS_DIR / "build_silver_span_labels.py"),
        "--span_ready_path",
        str(SPAN_READY_PATH),
        "--span_path",
        str(WINDOWS_PATH),
    ]


def build_span_dataset_cmd(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(ICR_SCRIPTS_DIR / "build_span_dataset.py"),
        "--span_ready_path",
        str(SPAN_READY_PATH),
        "--labeled_span_path",
        str(SILVER_PATH),
    ]


def build_train_baseline_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        str(ICR_SCRIPTS_DIR / "train_baseline_mlp.py"),
        "--dataset_path",
        str(DATASET_PATH),
        "--device",
        args.device,
    ]
    if args.n_splits is not None:
        cmd.extend(["--n_splits", str(args.n_splits)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.patience is not None:
        cmd.extend(["--patience", str(args.patience)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    return cmd


def build_train_discrepancy_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        str(ICR_SCRIPTS_DIR / "train_discrepancy.py"),
        "--dataset_path",
        str(DATASET_PATH),
    ]
    if args.n_splits is not None:
        cmd.extend(["--n_splits", str(args.n_splits)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.rf_estimators is not None:
        cmd.extend(["--rf_estimators", str(args.rf_estimators)])
    if args.max_iter is not None:
        cmd.extend(["--max_iter", str(args.max_iter)])
    return cmd


def metrics_exist(path: Path) -> bool:
    return path.exists() and any(path.glob("*.metrics.json"))


def _metadata_path(step_key: str) -> Path:
    return RUN_METADATA_DIR / f"{step_key}.json"


def _load_step_metadata(step_key: str) -> dict[str, object] | None:
    path = _metadata_path(step_key)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_step_metadata(
    step_key: str,
    *,
    config: dict[str, object],
    outputs: list[Path],
) -> None:
    payload = {
        "step": step_key,
        "config": config,
        "outputs": [str(path) for path in outputs],
    }
    path = _metadata_path(step_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _paths_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _step_skip_state(
    step_key: str,
    *,
    config: dict[str, object],
    outputs: list[Path],
    metrics_dir: Path | None = None,
) -> tuple[bool, str]:
    outputs_ready = metrics_exist(metrics_dir) if metrics_dir is not None else _paths_exist(outputs)
    if not outputs_ready:
        return False, "required outputs are missing"

    metadata = _load_step_metadata(step_key)
    if metadata is None:
        return False, "run metadata is missing"
    if metadata.get("config") != config:
        return False, "run metadata does not match the requested configuration"
    return True, "outputs exist and configuration matches"


def _normalize_window_sizes(window_sizes: str) -> list[int]:
    return [int(part.strip()) for part in window_sizes.split(",") if part.strip()]


def _prepare_span_ready_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "input_path": str(ICR_INPUT_JSONL),
        "qa_data_path": str(QA_DATA_JSON),
        "max_samples": args.max_samples,
    }


def _tokenizer_windows_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "input_path": str(SPAN_READY_PATH),
        "max_samples": args.max_samples,
        "window_sizes": _normalize_window_sizes(args.window_sizes),
    }


def _silver_labels_config() -> dict[str, object]:
    return {
        "span_ready_path": str(SPAN_READY_PATH),
        "span_path": str(WINDOWS_PATH),
        "positive_threshold": SILVER_POSITIVE_THRESHOLD,
        "negative_threshold": SILVER_NEGATIVE_THRESHOLD,
    }


def _span_dataset_config() -> dict[str, object]:
    return {
        "span_ready_path": str(SPAN_READY_PATH),
        "labeled_span_path": str(SILVER_PATH),
        "pooling": DATASET_POOLING,
    }


def _baseline_train_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_path": str(DATASET_PATH),
        "device": args.device,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "seed": args.seed,
    }


def _discrepancy_train_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_path": str(DATASET_PATH),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "rf_estimators": args.rf_estimators,
        "max_iter": args.max_iter,
    }


def sync_tree_contents(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_path in src_dir.rglob("*"):
        if not src_path.is_file():
            continue
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied += 1
    return copied


def print_metrics_summary(results_root: Path) -> None:
    metrics_files = sorted(results_root.rglob("*.metrics.json"))
    if not metrics_files:
        print("No metrics files found in copied baseline results.")
        return

    print("\nMetrics summary:")
    for metrics_path in metrics_files:
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        sample_max = metrics.get("sample_level", {}).get("max", {})
        auroc = sample_max.get("AUROC_mean")
        rel_path = metrics_path.relative_to(results_root)
        if auroc is None:
            print(f"  {rel_path}: sample(max) AUROC=N/A")
        else:
            print(f"  {rel_path}: sample(max) AUROC={auroc:.4f}")


def main() -> None:
    args = parse_args()
    verify_required_inputs()

    baseline_output_dir = ICR_RESULTS_DIR / DATASET_PATH.stem / "baseline_mlp"
    discrepancy_output_dir = ICR_RESULTS_DIR / DATASET_PATH.stem / "discrepancy"
    span_ready_outputs = [SPAN_READY_PATH, SPAN_READY_PATH.with_suffix(".summary.json")]
    tokenizer_window_outputs = [WINDOWS_PATH, WINDOWS_PATH.with_suffix(".summary.json")]
    silver_label_outputs = [SILVER_PATH, SILVER_PATH.with_suffix(".summary.json")]
    dataset_outputs = [DATASET_PATH, DATASET_PATH.with_suffix(".summary.json")]
    prepare_skip, prepare_reason = _step_skip_state(
        "prepare_span_ready",
        config=_prepare_span_ready_config(args),
        outputs=span_ready_outputs,
    )
    tokenizer_skip, tokenizer_reason = _step_skip_state(
        "build_tokenizer_windows",
        config=_tokenizer_windows_config(args),
        outputs=tokenizer_window_outputs,
    )
    silver_skip, silver_reason = _step_skip_state(
        "build_silver_labels",
        config=_silver_labels_config(),
        outputs=silver_label_outputs,
    )
    dataset_skip, dataset_reason = _step_skip_state(
        "build_span_dataset",
        config=_span_dataset_config(),
        outputs=dataset_outputs,
    )
    baseline_skip, baseline_reason = _step_skip_state(
        "train_baseline_mlp",
        config=_baseline_train_config(args),
        outputs=[],
        metrics_dir=baseline_output_dir,
    )
    discrepancy_skip, discrepancy_reason = _step_skip_state(
        "train_discrepancy",
        config=_discrepancy_train_config(args),
        outputs=[],
        metrics_dir=discrepancy_output_dir,
    )

    print(f"ICR repo: {ICR_REPO_ROOT}")
    print(f"ICR input: {ICR_INPUT_JSONL}")
    print(f"QA data: {QA_DATA_JSON}")
    print(f"Tokenizer-window dataset target: {DATASET_PATH}")
    print(f"Baseline results target: {BASELINE_RESULTS_DIR}")

    if run_step(
        build_prepare_span_ready_cmd(args),
        "Prepare span-ready data",
        skip=args.skip_existing and prepare_skip,
        skip_path=SPAN_READY_PATH,
        skip_detail=prepare_reason if args.skip_existing and prepare_skip else None,
    ):
        _save_step_metadata(
            "prepare_span_ready",
            config=_prepare_span_ready_config(args),
            outputs=span_ready_outputs,
        )
    if run_step(
        build_tokenizer_windows_cmd(args),
        "Build tokenizer windows",
        skip=args.skip_existing and tokenizer_skip,
        skip_path=WINDOWS_PATH,
        skip_detail=tokenizer_reason if args.skip_existing and tokenizer_skip else None,
    ):
        _save_step_metadata(
            "build_tokenizer_windows",
            config=_tokenizer_windows_config(args),
            outputs=tokenizer_window_outputs,
        )
    if run_step(
        build_silver_labels_cmd(args),
        "Build silver span labels",
        skip=args.skip_existing and silver_skip,
        skip_path=SILVER_PATH,
        skip_detail=silver_reason if args.skip_existing and silver_skip else None,
    ):
        _save_step_metadata(
            "build_silver_labels",
            config=_silver_labels_config(),
            outputs=silver_label_outputs,
        )
    if run_step(
        build_span_dataset_cmd(args),
        "Build span dataset",
        skip=args.skip_existing and dataset_skip,
        skip_path=DATASET_PATH,
        skip_detail=dataset_reason if args.skip_existing and dataset_skip else None,
    ):
        _save_step_metadata(
            "build_span_dataset",
            config=_span_dataset_config(),
            outputs=dataset_outputs,
        )
    if run_step(
        build_train_baseline_cmd(args),
        "Train baseline MLP",
        skip=args.skip_existing and baseline_skip,
        skip_path=baseline_output_dir,
        skip_detail=baseline_reason if args.skip_existing and baseline_skip else None,
    ):
        _save_step_metadata(
            "train_baseline_mlp",
            config=_baseline_train_config(args),
            outputs=[baseline_output_dir],
        )
    if run_step(
        build_train_discrepancy_cmd(args),
        "Train discrepancy models",
        skip=args.skip_existing and discrepancy_skip,
        skip_path=discrepancy_output_dir,
        skip_detail=discrepancy_reason if args.skip_existing and discrepancy_skip else None,
    ):
        _save_step_metadata(
            "train_discrepancy",
            config=_discrepancy_train_config(args),
            outputs=[discrepancy_output_dir],
        )

    copied_baseline = sync_tree_contents(baseline_output_dir, BASELINE_RESULTS_DIR / "baseline_mlp")
    copied_discrepancy = sync_tree_contents(discrepancy_output_dir, BASELINE_RESULTS_DIR / "discrepancy")

    print(f"\nCopied {copied_baseline} baseline files to {BASELINE_RESULTS_DIR / 'baseline_mlp'}")
    print(f"Copied {copied_discrepancy} discrepancy files to {BASELINE_RESULTS_DIR / 'discrepancy'}")
    print(f"\nBaseline pipeline complete. Results saved to {BASELINE_RESULTS_DIR}")
    print_metrics_summary(BASELINE_RESULTS_DIR)


if __name__ == "__main__":
    main()
