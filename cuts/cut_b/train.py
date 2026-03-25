from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()

from cut_b.features import extract_mismatch_features
from shared.data_loader import load_combined_span_dataset
from shared.eval_utils import (
    aggregate_sample_predictions,
    build_group_folds,
    dump_json,
    ensure_parent_dir,
    evaluate_binary_predictions,
    print_metrics_summary,
    summarize_metric_dicts,
    write_jsonl,
)
from spanlab.dependencies import require_sklearn, require_torch


def _check_binary_training_labels(train_labels: np.ndarray, model_name: str, fold_id: int) -> None:
    unique_labels = np.unique(train_labels)
    if unique_labels.size < 2:
        raise ValueError(
            f"{model_name} fold {fold_id + 1} has only one training class in silver labels: {unique_labels.tolist()}"
        )


def _prediction_probability(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)
    scores = np.asarray(model.decision_function(features), dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)


def _base_output_payload(
    rows: Sequence[dict[str, Any]],
    probabilities: np.ndarray,
    fold_assignments: np.ndarray,
    feature_set_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row, probability, fold_id in zip(rows, probabilities, fold_assignments):
        payload.append(
            {
                "feature_set": feature_set_name,
                "model": model_name,
                "span_id": row.get("span_id"),
                "sample_id": row.get("sample_id"),
                "route": row.get("route"),
                "span_type": row.get("span_type"),
                "sample_label": int(row["sample_label"]),
                "silver_label": row.get("silver_label"),
                "silver_confidence": row.get("silver_confidence"),
                "is_labeled": bool(row.get("is_labeled")),
                "fold": int(fold_id),
                "probability": None if np.isnan(probability) else float(probability),
            }
        )
    return payload


def _build_sklearn_model_factories(seed: int) -> dict[str, Callable[[], Any]]:
    require_sklearn()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
    }


def _fit_torch_model(
    model_factory: Callable[[int], Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val_labeled: np.ndarray,
    y_val_labeled: np.ndarray,
    x_val_all: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    weight_decay: float,
) -> np.ndarray:
    torch = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    criterion = torch.nn.BCELoss()
    model = model_factory(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(2, patience // 2), factor=0.5)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_val_labeled), torch.FloatTensor(y_val_labeled)),
        batch_size=batch_size,
        shuffle=False,
    )
    predict_loader = DataLoader(torch.FloatTensor(x_val_all), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x).view(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        total_val_rows = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x).view(-1)
                loss = criterion(pred, batch_y)
                total_val_loss += float(loss.item()) * len(batch_y)
                total_val_rows += len(batch_y)
        val_loss = total_val_loss / max(total_val_rows, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError("Torch training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for batch_x in predict_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).view(-1)
            outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    features: np.ndarray
    feature_names: list[str]


def _usable_icr_vectors(icr_vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(icr_vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected ICR vectors with shape [N, n_layers], got {vectors.shape}.")
    if vectors.shape[1] < 27:
        raise ValueError(f"Expected at least 27 ICR layers, got {vectors.shape[1]}.")
    return vectors[:, :27].astype(np.float32)


def _make_feature_sets(icr_vectors: np.ndarray, entropy_vectors: np.ndarray) -> list[FeatureSet]:
    icr_vectors = _usable_icr_vectors(icr_vectors)
    mismatch_features, mismatch_names = extract_mismatch_features(entropy_vectors)
    icr_names = [f"icr_layer_{index:02d}" for index in range(icr_vectors.shape[1])]
    combined_names = icr_names + mismatch_names

    return [
        FeatureSet(name="icr_only", features=icr_vectors.astype(np.float32), feature_names=icr_names),
        FeatureSet(name="mismatch_only", features=mismatch_features.astype(np.float32), feature_names=mismatch_names),
        FeatureSet(
            name="icr_plus_mismatch",
            features=np.concatenate([icr_vectors, mismatch_features], axis=1).astype(np.float32),
            feature_names=combined_names,
        ),
    ]


def _run_sklearn_models(
    rows: Sequence[dict[str, Any]],
    features: np.ndarray,
    feature_names: list[str],
    silver_labels: np.ndarray,
    sample_ids: np.ndarray,
    sample_labels: np.ndarray,
    output_dir: Path,
    feature_set_name: str,
    n_splits: int,
    seed: int,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    folds = build_group_folds(sample_ids, sample_labels, n_splits=n_splits, seed=seed)
    model_factories = _build_sklearn_model_factories(seed=seed)
    ensure_parent_dir(output_dir / "placeholder")

    results: dict[str, dict[str, Any]] = {}
    for model_name, model_factory in model_factories.items():
        print(f"\n[{feature_set_name}] Training {model_name}")
        probabilities = np.full(len(rows), np.nan, dtype=np.float32)
        fold_assignments = np.full(len(rows), -1, dtype=np.int32)
        span_metrics: list[dict[str, float]] = []
        sample_metrics = {"max": [], "topk_mean": [], "noisy_or": []}

        for fold_id, (train_samples, val_samples) in enumerate(folds):
            train_mask = np.isin(sample_ids, list(train_samples)) & (silver_labels >= 0)
            val_mask_all = np.isin(sample_ids, list(val_samples))
            val_mask_labeled = val_mask_all & (silver_labels >= 0)

            _check_binary_training_labels(silver_labels[train_mask], model_name=model_name, fold_id=fold_id)
            model = model_factory()
            model.fit(features[train_mask], silver_labels[train_mask])

            fold_probs = _prediction_probability(model, features[val_mask_all])
            probabilities[val_mask_all] = fold_probs
            fold_assignments[val_mask_all] = fold_id

            if val_mask_labeled.any():
                span_metrics.append(
                    evaluate_binary_predictions(
                        silver_labels[val_mask_labeled],
                        probabilities[val_mask_labeled],
                    )
                )

            fold_rows = [rows[index] for index in np.where(val_mask_all)[0]]
            aggregated = aggregate_sample_predictions(fold_rows, fold_probs, top_k=top_k)
            for mode, payload in aggregated.items():
                sample_metrics[mode].append(evaluate_binary_predictions(payload["labels"], payload["probs"]))

        metrics_payload = {
            "family": "cut_b_sklearn",
            "feature_set": feature_set_name,
            "model": model_name,
            "feature_names": feature_names,
            "feature_dim": len(feature_names),
            "n_rows": len(rows),
            "n_labeled_rows": int((silver_labels >= 0).sum()),
            "span_level": summarize_metric_dicts(span_metrics),
            "sample_level": {mode: summarize_metric_dicts(metric_rows) for mode, metric_rows in sample_metrics.items()},
        }
        dump_json(output_dir / f"{model_name}.metrics.json", metrics_payload)
        write_jsonl(
            output_dir / f"{model_name}.oof_predictions.jsonl",
            _base_output_payload(rows, probabilities, fold_assignments, feature_set_name, model_name),
        )
        results[model_name] = metrics_payload
        print_metrics_summary(metrics_payload, prefix=f"{feature_set_name} / {model_name}")

    return results


def _run_torch_models(
    rows: Sequence[dict[str, Any]],
    features: np.ndarray,
    feature_names: list[str],
    silver_labels: np.ndarray,
    sample_ids: np.ndarray,
    sample_labels: np.ndarray,
    output_dir: Path,
    feature_set_name: str,
    n_splits: int,
    seed: int,
    top_k: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    weight_decay: float,
) -> dict[str, dict[str, Any]]:
    torch = require_torch()
    torch.manual_seed(seed)

    from spanlab.models import BaselineMLP

    folds = build_group_folds(sample_ids, sample_labels, n_splits=n_splits, seed=seed)
    ensure_parent_dir(output_dir / "placeholder")
    model_factories: dict[str, Callable[[int], Any]] = {"baseline_mlp": lambda input_dim: BaselineMLP(input_dim=input_dim)}

    results: dict[str, dict[str, Any]] = {}
    for model_name, model_factory in model_factories.items():
        print(f"\n[{feature_set_name}] Training {model_name}")
        probabilities = np.full(len(rows), np.nan, dtype=np.float32)
        fold_assignments = np.full(len(rows), -1, dtype=np.int32)
        span_metrics: list[dict[str, float]] = []
        sample_metrics = {"max": [], "topk_mean": [], "noisy_or": []}

        for fold_id, (train_samples, val_samples) in enumerate(folds):
            train_mask = np.isin(sample_ids, list(train_samples)) & (silver_labels >= 0)
            val_mask_all = np.isin(sample_ids, list(val_samples))
            val_mask_labeled = val_mask_all & (silver_labels >= 0)

            _check_binary_training_labels(silver_labels[train_mask], model_name=model_name, fold_id=fold_id)

            train_x = features[train_mask]
            val_x_all = features[val_mask_all]
            val_x_labeled = features[val_mask_labeled]
            train_y = silver_labels[train_mask].astype(np.float32)
            val_y_labeled = silver_labels[val_mask_labeled].astype(np.float32)

            mean = train_x.mean(axis=0, keepdims=True)
            std = train_x.std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)

            train_x = (train_x - mean) / std
            val_x_all = (val_x_all - mean) / std
            val_x_labeled = (val_x_labeled - mean) / std

            fold_probs = _fit_torch_model(
                model_factory=model_factory,
                x_train=train_x,
                y_train=train_y,
                x_val_labeled=val_x_labeled,
                y_val_labeled=val_y_labeled,
                x_val_all=val_x_all,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience,
                weight_decay=weight_decay,
            )

            probabilities[val_mask_all] = fold_probs
            fold_assignments[val_mask_all] = fold_id

            if val_mask_labeled.any():
                span_metrics.append(
                    evaluate_binary_predictions(
                        silver_labels[val_mask_labeled],
                        probabilities[val_mask_labeled],
                    )
                )

            fold_rows = [rows[index] for index in np.where(val_mask_all)[0]]
            aggregated = aggregate_sample_predictions(fold_rows, fold_probs, top_k=top_k)
            for mode, payload in aggregated.items():
                sample_metrics[mode].append(evaluate_binary_predictions(payload["labels"], payload["probs"]))

        metrics_payload = {
            "family": "cut_b_torch",
            "feature_set": feature_set_name,
            "model": model_name,
            "feature_names": feature_names,
            "feature_dim": len(feature_names),
            "n_rows": len(rows),
            "n_labeled_rows": int((silver_labels >= 0).sum()),
            "span_level": summarize_metric_dicts(span_metrics),
            "sample_level": {mode: summarize_metric_dicts(metric_rows) for mode, metric_rows in sample_metrics.items()},
        }
        dump_json(output_dir / f"{model_name}.metrics.json", metrics_payload)
        write_jsonl(
            output_dir / f"{model_name}.oof_predictions.jsonl",
            _base_output_payload(rows, probabilities, fold_assignments, feature_set_name, model_name),
        )
        results[model_name] = metrics_payload
        print_metrics_summary(metrics_payload, prefix=f"{feature_set_name} / {model_name}")

    return results


def _metric_value(metrics: dict[str, Any], section: str, key: str, mode: str | None = None) -> float | None:
    if section == "sample_level":
        value = metrics.get(section, {}).get(mode or "max", {}).get(key)
    else:
        value = metrics.get(section, {}).get(key)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _build_comparison_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'Feature Set':<18} {'Model':<20} {'Span AUROC':>12} "
        f"{'Sample AUROC':>12} {'Sample AUPRC':>12} {'Sample F1':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['feature_set']:<18} {row['model']:<20} "
            f"{_format_metric(row['span_auroc']):>12} "
            f"{_format_metric(row['sample_auroc']):>12} "
            f"{_format_metric(row['sample_auprc']):>12} "
            f"{_format_metric(row['sample_f1']):>10}"
        )
    return "\n".join(lines)


def run_mismatch_experiment(combined_dataset_path: Path, output_dir: Path, device: str = "cpu") -> dict[str, Any]:
    """Train Cut B classifiers on ICR, mismatch, and joint feature sets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        rows,
        icr_vectors,
        entropy_vectors,
        _delta_entropy_vectors,
        silver_labels,
        sample_ids,
        sample_labels,
    ) = load_combined_span_dataset(Path(combined_dataset_path))

    feature_sets = _make_feature_sets(icr_vectors=icr_vectors, entropy_vectors=entropy_vectors)
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    comparison_rows: list[dict[str, Any]] = []

    for feature_set in feature_sets:
        feature_dir = output_dir / feature_set.name
        sklearn_dir = feature_dir / "sklearn"
        torch_dir = feature_dir / "torch"

        sklearn_results = _run_sklearn_models(
            rows=rows,
            features=feature_set.features,
            feature_names=feature_set.feature_names,
            silver_labels=silver_labels,
            sample_ids=sample_ids,
            sample_labels=sample_labels,
            output_dir=sklearn_dir,
            feature_set_name=feature_set.name,
            n_splits=5,
            seed=42,
            top_k=3,
        )
        torch_results = _run_torch_models(
            rows=rows,
            features=feature_set.features,
            feature_names=feature_set.feature_names,
            silver_labels=silver_labels,
            sample_ids=sample_ids,
            sample_labels=sample_labels,
            output_dir=torch_dir,
            feature_set_name=feature_set.name,
            n_splits=5,
            seed=42,
            top_k=3,
            device=device,
            epochs=50,
            batch_size=64,
            learning_rate=1e-3,
            patience=8,
            weight_decay=1e-4,
        )

        all_results[feature_set.name] = {"sklearn": sklearn_results, "torch": torch_results}

        for model_group in (sklearn_results, torch_results):
            for model_name, metrics in model_group.items():
                comparison_rows.append(
                    {
                        "feature_set": feature_set.name,
                        "model": model_name,
                        "span_auroc": _metric_value(metrics, "span_level", "AUROC_mean"),
                        "sample_auroc": _metric_value(metrics, "sample_level", "AUROC_mean", mode="max"),
                        "sample_auprc": _metric_value(metrics, "sample_level", "AUPRC_mean", mode="max"),
                        "sample_f1": _metric_value(metrics, "sample_level", "F1_mean", mode="max"),
                    }
                )

    comparison_table = _build_comparison_table(comparison_rows)
    print("\nCut B comparison table")
    print(comparison_table)

    summary = {
        "combined_dataset_path": str(combined_dataset_path),
        "device": device,
        "results": all_results,
        "comparison_rows": comparison_rows,
        "comparison_table": comparison_table,
    }
    dump_json(output_dir / "training_summary.json", summary)
    (output_dir / "comparison_table.txt").write_text(comparison_table + "\n", encoding="utf-8")
    return summary


__all__ = ["run_mismatch_experiment"]
