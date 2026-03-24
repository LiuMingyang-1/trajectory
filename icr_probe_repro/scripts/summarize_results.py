#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.io_utils import dump_json, ensure_parent_dir
from spanlab.paths import (
    RESULTS_DIR,
    default_results_summary_csv_path,
    default_results_summary_json_path,
    default_results_summary_md_path,
)
from spanlab.visualization import load_metric_records, prettify_dataset_name, prettify_model_name


def span_metric(payload: Dict, metric_name: str) -> float:
    return float(payload.get("span_level", {}).get(metric_name, 0.0))


def sample_metric(payload: Dict, mode: str, metric_name: str) -> float:
    return float(payload.get("sample_level", {}).get(mode, {}).get(metric_name, 0.0))


def build_summary_rows(results_root: Path) -> List[Dict]:
    rows = []
    for record in load_metric_records(results_root):
        payload = record["payload"]
        rows.append(
            {
                "dataset_name": record["dataset_name"],
                "dataset_label": prettify_dataset_name(record["dataset_name"]),
                "family_name": record["family_name"],
                "model_name": record["model_name"],
                "model_label": prettify_model_name(record["model_name"]),
                "metrics_path": record["metrics_path"],
                "n_rows": int(payload.get("n_rows", 0)),
                "n_labeled_rows": int(payload.get("n_labeled_rows", 0)),
                "span_auroc_mean": span_metric(payload, "AUROC_mean"),
                "span_auprc_mean": span_metric(payload, "AUPRC_mean"),
                "span_f1_mean": span_metric(payload, "F1_mean"),
                "sample_max_auroc_mean": sample_metric(payload, "max", "AUROC_mean"),
                "sample_topk_auroc_mean": sample_metric(payload, "topk_mean", "AUROC_mean"),
                "sample_noisy_or_auroc_mean": sample_metric(payload, "noisy_or", "AUROC_mean"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["dataset_name"],
            -row["sample_noisy_or_auroc_mean"],
            -row["span_auroc_mean"],
            row["model_name"],
        )
    )
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_parent_dir(path)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: List[Dict]) -> None:
    ensure_parent_dir(path)
    lines = [
        "# Experiment Summary",
        "",
        "| Dataset | Model | Span AUROC | Span AUPRC | Sample AUROC (Noisy-Or) | Sample AUROC (Max) | Sample AUROC (Top-k) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset_label"],
                    row["model_label"],
                    f"{row['span_auroc_mean']:.4f}",
                    f"{row['span_auprc_mean']:.4f}",
                    f"{row['sample_noisy_or_auroc_mean']:.4f}",
                    f"{row['sample_max_auroc_mean']:.4f}",
                    f"{row['sample_topk_auroc_mean']:.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all metric files under results/.")
    parser.add_argument("--results_root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output_json", type=Path, default=default_results_summary_json_path())
    parser.add_argument("--output_csv", type=Path, default=default_results_summary_csv_path())
    parser.add_argument("--output_md", type=Path, default=default_results_summary_md_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_summary_rows(args.results_root)
    if not rows:
        raise ValueError(f"No metric files found under {args.results_root}")

    dump_json(args.output_json, rows)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)

    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.output_md}")


if __name__ == "__main__":
    main()
