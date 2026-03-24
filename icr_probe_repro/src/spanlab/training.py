from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aggregation import aggregate_sample_predictions
from .dependencies import require_sklearn, require_torch
from .evaluation import build_group_folds, evaluate_binary_predictions, summarize_metric_dicts
from .io_utils import dump_json, ensure_parent_dir, read_jsonl, write_jsonl


FeatureBuilder = Optional[Callable[[np.ndarray], Tuple[np.ndarray, Optional[List[str]]]]]


def load_span_dataset(dataset_path: Path) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset_path}")

    vectors = np.asarray([row["span_vector"] for row in rows], dtype=np.float32)
    silver_labels = np.asarray(
        [-1 if row.get("silver_label") is None else int(row["silver_label"]) for row in rows],
        dtype=np.int32,
    )
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=object)
    sample_labels = np.asarray([int(row["sample_label"]) for row in rows], dtype=np.int32)
    return rows, vectors, silver_labels, sample_ids, sample_labels


def resolve_features(raw_vectors: np.ndarray, feature_builder: FeatureBuilder) -> Tuple[np.ndarray, Optional[List[str]]]:
    if feature_builder is None:
        return raw_vectors.astype(np.float32), None
    built = feature_builder(raw_vectors)
    if isinstance(built, tuple):
        features, names = built
        return np.asarray(features, dtype=np.float32), names
    return np.asarray(built, dtype=np.float32), None


def _prediction_probability(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)
    scores = np.asarray(model.decision_function(features), dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-scores))


def _base_output_payload(
    rows: Sequence[Dict],
    probabilities: np.ndarray,
    fold_assignments: np.ndarray,
) -> List[Dict]:
    payload = []
    for row, probability, fold_id in zip(rows, probabilities, fold_assignments):
        payload.append(
            {
                "span_id": row["span_id"],
                "sample_id": row["sample_id"],
                "route": row["route"],
                "span_type": row["span_type"],
                "sample_label": int(row["sample_label"]),
                "silver_label": row.get("silver_label"),
                "silver_confidence": row.get("silver_confidence"),
                "is_labeled": bool(row.get("is_labeled")),
                "fold": int(fold_id),
                "probability": None if np.isnan(probability) else float(probability),
            }
        )
    return payload


def _check_binary_training_labels(train_labels: np.ndarray, model_name: str, fold_id: int) -> None:
    unique_labels = np.unique(train_labels)
    if unique_labels.size < 2:
        raise ValueError(
            f"{model_name} fold {fold_id + 1} has only one training class in silver labels: {unique_labels.tolist()}"
        )


def run_sklearn_family(
    dataset_path: Path,
    output_dir: Path,
    family_name: str,
    model_factories: Dict[str, Callable[[], object]],
    feature_builder: FeatureBuilder = None,
    n_splits: int = 5,
    seed: int = 42,
    top_k: int = 3,
) -> Dict[str, Dict]:
    require_sklearn()

    rows, raw_vectors, silver_labels, sample_ids, sample_labels = load_span_dataset(dataset_path)
    features, feature_names = resolve_features(raw_vectors, feature_builder)
    folds = build_group_folds(sample_ids, sample_labels, n_splits=n_splits, seed=seed)

    ensure_parent_dir(output_dir / "placeholder")
    family_results: Dict[str, Dict] = {}

    for model_name, model_factory in model_factories.items():
        print(f"\n[{family_name}] Training {model_name}")
        probabilities = np.full(len(rows), np.nan, dtype=np.float32)
        fold_assignments = np.full(len(rows), -1, dtype=np.int32)
        span_metrics: List[Dict[str, float]] = []
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
            "family": family_name,
            "model": model_name,
            "dataset_path": str(dataset_path),
            "feature_names": feature_names,
            "n_rows": len(rows),
            "n_labeled_rows": int((silver_labels >= 0).sum()),
            "span_level": summarize_metric_dicts(span_metrics),
            "sample_level": {mode: summarize_metric_dicts(metric_rows) for mode, metric_rows in sample_metrics.items()},
        }

        metrics_path = output_dir / f"{model_name}.metrics.json"
        prediction_path = output_dir / f"{model_name}.oof_predictions.jsonl"
        dump_json(metrics_path, metrics_payload)
        write_jsonl(prediction_path, _base_output_payload(rows, probabilities, fold_assignments))

        family_results[model_name] = metrics_payload
        print(
            f"  span AUROC={metrics_payload['span_level'].get('AUROC_mean', float('nan')):.4f} | "
            f"sample(max) AUROC={metrics_payload['sample_level']['max'].get('AUROC_mean', float('nan')):.4f}"
        )

    return family_results


def _fit_torch_model(
    model_factory: Callable[[int], object],
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
):
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


def run_torch_family(
    dataset_path: Path,
    output_dir: Path,
    family_name: str,
    model_factories: Dict[str, Callable[[int], object]],
    feature_builder: FeatureBuilder = None,
    n_splits: int = 5,
    seed: int = 42,
    top_k: int = 3,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 8,
    weight_decay: float = 1e-4,
) -> Dict[str, Dict]:
    torch = require_torch()
    torch.manual_seed(seed)

    rows, raw_vectors, silver_labels, sample_ids, sample_labels = load_span_dataset(dataset_path)
    features, feature_names = resolve_features(raw_vectors, feature_builder)
    folds = build_group_folds(sample_ids, sample_labels, n_splits=n_splits, seed=seed)

    ensure_parent_dir(output_dir / "placeholder")
    family_results: Dict[str, Dict] = {}

    for model_name, model_factory in model_factories.items():
        print(f"\n[{family_name}] Training {model_name}")
        probabilities = np.full(len(rows), np.nan, dtype=np.float32)
        fold_assignments = np.full(len(rows), -1, dtype=np.int32)
        span_metrics: List[Dict[str, float]] = []
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
            "family": family_name,
            "model": model_name,
            "dataset_path": str(dataset_path),
            "feature_names": feature_names,
            "n_rows": len(rows),
            "n_labeled_rows": int((silver_labels >= 0).sum()),
            "span_level": summarize_metric_dicts(span_metrics),
            "sample_level": {mode: summarize_metric_dicts(metric_rows) for mode, metric_rows in sample_metrics.items()},
        }

        metrics_path = output_dir / f"{model_name}.metrics.json"
        prediction_path = output_dir / f"{model_name}.oof_predictions.jsonl"
        dump_json(metrics_path, metrics_payload)
        write_jsonl(prediction_path, _base_output_payload(rows, probabilities, fold_assignments))

        family_results[model_name] = metrics_payload
        print(
            f"  span AUROC={metrics_payload['span_level'].get('AUROC_mean', float('nan')):.4f} | "
            f"sample(max) AUROC={metrics_payload['sample_level']['max'].get('AUROC_mean', float('nan')):.4f}"
        )

    return family_results
