#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LAB_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.dependencies import require_sklearn
from spanlab.features import extract_change_point_features
from spanlab.paths import RESULTS_DIR
from spanlab.training import run_sklearn_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train change-point span-level models.")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf_estimators", type=int, default=200)
    parser.add_argument("--max_iter", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_sklearn()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    output_dir = args.output_dir or (RESULTS_DIR / args.dataset_path.stem / "change_point")
    run_sklearn_family(
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        family_name="change_point",
        model_factories={
            "LogisticRegression": lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=args.max_iter, random_state=args.seed),
            ),
            "RandomForest": lambda: RandomForestClassifier(
                n_estimators=args.rf_estimators,
                random_state=args.seed,
            ),
        },
        feature_builder=extract_change_point_features,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
