#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.paths import RESULTS_DIR
from spanlab.training import run_torch_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the span-level baseline MLP.")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from spanlab.models import BaselineMLP

    output_dir = args.output_dir or (RESULTS_DIR / args.dataset_path.stem / "baseline_mlp")
    run_torch_family(
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        family_name="baseline_mlp",
        model_factories={"BaselineMLP": lambda input_dim: BaselineMLP(input_dim=input_dim)},
        n_splits=args.n_splits,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
