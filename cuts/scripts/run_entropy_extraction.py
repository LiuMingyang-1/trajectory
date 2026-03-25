#!/usr/bin/env python3
"""CLI entry point for extracting per-layer logit entropy from model hidden states."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPTS_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ENTROPY_JSONL, ICR_INPUT_JSONL


def summarize_output(path: Path) -> tuple[int, dict[str, object] | None]:
    from shared.entropy import entropy_summary_stats

    count = 0
    matrices: list[np.ndarray] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            matrices.append(np.asarray(row["entropy_scores"], dtype=np.float32))
            count += 1

    if not matrices:
        return count, None

    non_empty = [matrix.reshape(-1) for matrix in matrices if matrix.size > 0]
    if not non_empty:
        return count, entropy_summary_stats(matrices[0])

    flat_values = np.concatenate(non_empty)
    return count, {
        "min": float(flat_values.min()),
        "max": float(flat_values.max()),
        "mean": float(flat_values.mean()),
        "std": float(flat_values.std()),
    }


def main() -> None:
    try:
        from shared.inference import main as inference_main
        from shared.inference import parse_args
    except ModuleNotFoundError as exc:
        print(
            "ERROR: shared.inference dependencies are missing. "
            f"Import failed for module '{exc.name}'."
        )
        sys.exit(1)

    args = parse_args()
    input_path = Path(args.icr_input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print(f"ERROR: ICR input not found at {input_path}")
        print(f"Default expected location: {ICR_INPUT_JSONL}")
        sys.exit(1)

    print(f"ICR input: {input_path}")
    print(f"Entropy output: {output_path}")

    inference_main()

    if not output_path.exists():
        print(f"ERROR: Entropy output was not created at {output_path}")
        sys.exit(1)

    count, stats = summarize_output(output_path)
    print(f"\nPost-check: wrote {count} records to {output_path}")
    if stats is None:
        print("Entropy stats: no non-empty records found in the output file.")
        return

    print(f"Entropy stats: {stats}")


if __name__ == "__main__":
    main()
