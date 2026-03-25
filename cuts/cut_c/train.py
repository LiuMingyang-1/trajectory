"""Training utilities for Cut C adaptive gated probes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()

from cut_c.gating import ArrayScaler, CutCFeatureBundle, GatedProbe, build_cut_c_feature_bundle
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
from spanlab.dependencies import require_torch


TOP_K = 3
TORCH_EPOCHS = 50
STAGED_EXPERT_EPOCHS = 40
STAGED_GATE_EPOCHS = 30
TORCH_BATCH_SIZE = 64
TORCH_LEARNING_RATE = 1e-3
TORCH_PATIENCE = 8
TORCH_WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class FoldPrediction:
    probabilities: np.ndarray
    gate_values: np.ndarray | None = None
    change_probabilities: np.ndarray | None = None
    icr_probabilities: np.ndarray | None = None


def _seed_torch(seed: int) -> None:
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_binary_training_labels(train_labels: np.ndarray, model_name: str, fold_id: int) -> None:
    unique_labels = np.unique(train_labels)
    if unique_labels.size < 2:
        raise ValueError(
            f"{model_name} fold {fold_id + 1} has only one training class in silver labels: {unique_labels.tolist()}"
        )


def _clone_state_dict(module: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _should_drop_last(n_rows: int, batch_size: int) -> bool:
    return n_rows > 1 and (n_rows % batch_size == 1)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(array.size),
        "mean": _safe_float(array.mean()),
        "std": _safe_float(array.std(ddof=0)),
        "median": _safe_float(np.median(array)),
        "min": _safe_float(array.min()),
        "max": _safe_float(array.max()),
    }


def _metric_value(metrics: dict[str, Any], section: str, key: str, mode: str | None = None) -> float | None:
    if section == "sample_level":
        value = metrics.get(section, {}).get(mode or "max", {}).get(key)
    else:
        value = metrics.get(section, {}).get(key)
    return _safe_float(value)


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _build_comparison_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'Feature Set':<22} {'Model':<18} {'Span AUROC':>12} "
        f"{'Sample AUROC':>12} {'Sample AUPRC':>12} {'Sample F1':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['feature_set']:<22} {row['model']:<18} "
            f"{_format_metric(row['span_auroc']):>12} "
            f"{_format_metric(row['sample_auroc']):>12} "
            f"{_format_metric(row['sample_auprc']):>12} "
            f"{_format_metric(row['sample_f1']):>10}"
        )
    return "\n".join(lines)


def _base_output_payload(
    rows: Sequence[dict[str, Any]],
    probabilities: np.ndarray,
    fold_assignments: np.ndarray,
    feature_set_name: str,
    model_name: str,
    extra_fields: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, (row, probability, fold_id) in enumerate(zip(rows, probabilities, fold_assignments)):
        record = {
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
        if extra_fields is not None:
            for name, values in extra_fields.items():
                value = values[index]
                record[name] = None if np.isnan(value) else float(value)
        payload.append(record)
    return payload


def _fit_baseline_mlp(
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
    seed: int,
) -> np.ndarray:
    torch = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    from spanlab.models import BaselineMLP

    _seed_torch(seed)
    criterion = torch.nn.BCELoss()
    model = BaselineMLP(input_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(2, patience // 2), factor=0.5)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=_should_drop_last(len(x_train), batch_size),
    )
    val_loader = None
    if len(x_val_labeled):
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
        total_train_loss = 0.0
        total_train_rows = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x).view(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * len(batch_y)
            total_train_rows += len(batch_y)

        if val_loader is None:
            val_loss = total_train_loss / max(total_train_rows, 1)
        else:
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
            best_state = _clone_state_dict(model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError("Baseline MLP training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x in predict_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).view(-1)
            outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _fit_single_input_module(
    module: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val_labeled: np.ndarray,
    y_val_labeled: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    weight_decay: float,
) -> None:
    torch = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    criterion = torch.nn.BCELoss()
    module.to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(2, patience // 2), factor=0.5)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=_should_drop_last(len(x_train), batch_size),
    )
    val_loader = None
    if len(x_val_labeled):
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_val_labeled), torch.FloatTensor(y_val_labeled)),
            batch_size=batch_size,
            shuffle=False,
        )

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for _ in range(epochs):
        module.train()
        total_train_loss = 0.0
        total_train_rows = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = module(batch_x).view(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * len(batch_y)
            total_train_rows += len(batch_y)

        if val_loader is None:
            val_loss = total_train_loss / max(total_train_rows, 1)
        else:
            module.eval()
            total_val_loss = 0.0
            total_val_rows = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    pred = module(batch_x).view(-1)
                    loss = criterion(pred, batch_y)
                    total_val_loss += float(loss.item()) * len(batch_y)
                    total_val_rows += len(batch_y)
            val_loss = total_val_loss / max(total_val_rows, 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = _clone_state_dict(module)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError("Expert pretraining did not produce a best checkpoint.")

    module.load_state_dict(best_state)
    module.to(device)
    module.eval()


def _forward_gated_probe(
    model: GatedProbe,
    gate_x: Any,
    change_x: Any,
    icr_x: Any,
    *,
    experts_frozen: bool,
) -> dict[str, Any]:
    torch = require_torch()
    if not experts_frozen:
        return model(gate_x, change_x, icr_x)

    gate = model.gate_network(gate_x)
    with torch.no_grad():
        change_probability = model.change_expert(change_x)
        icr_probability = model.icr_expert(icr_x)
    probability = gate * change_probability + (1.0 - gate) * icr_probability
    return {
        "probability": probability,
        "gate": gate,
        "change_probability": change_probability,
        "icr_probability": icr_probability,
    }


def _fit_gated_probe(
    gate_train: np.ndarray,
    change_train: np.ndarray,
    icr_train: np.ndarray,
    y_train: np.ndarray,
    gate_val_labeled: np.ndarray,
    change_val_labeled: np.ndarray,
    icr_val_labeled: np.ndarray,
    y_val_labeled: np.ndarray,
    gate_val_all: np.ndarray,
    change_val_all: np.ndarray,
    icr_val_all: np.ndarray,
    device: str,
    learning_rate: float,
    batch_size: int,
    patience: int,
    weight_decay: float,
    joint_epochs: int,
    staged_expert_epochs: int,
    staged_gate_epochs: int,
    staged: bool,
    seed: int,
) -> FoldPrediction:
    torch = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    _seed_torch(seed)
    model = GatedProbe(
        gate_input_dim=gate_train.shape[1],
        change_input_dim=change_train.shape[1],
        icr_input_dim=icr_train.shape[1],
    ).to(device)

    if staged:
        _fit_single_input_module(
            model.change_expert,
            x_train=change_train,
            y_train=y_train,
            x_val_labeled=change_val_labeled,
            y_val_labeled=y_val_labeled,
            device=device,
            epochs=staged_expert_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            weight_decay=weight_decay,
        )
        _fit_single_input_module(
            model.icr_expert,
            x_train=icr_train,
            y_train=y_train,
            x_val_labeled=icr_val_labeled,
            y_val_labeled=y_val_labeled,
            device=device,
            epochs=staged_expert_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            weight_decay=weight_decay,
        )
        model.freeze_experts()
        optimizer = torch.optim.Adam(
            model.gate_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        n_epochs = staged_gate_epochs
    else:
        model.unfreeze_all()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        n_epochs = joint_epochs

    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(2, patience // 2), factor=0.5)
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(gate_train),
            torch.FloatTensor(change_train),
            torch.FloatTensor(icr_train),
            torch.FloatTensor(y_train),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=_should_drop_last(len(gate_train), batch_size),
    )
    val_loader = None
    if len(gate_val_labeled):
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(gate_val_labeled),
                torch.FloatTensor(change_val_labeled),
                torch.FloatTensor(icr_val_labeled),
                torch.FloatTensor(y_val_labeled),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    predict_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(gate_val_all),
            torch.FloatTensor(change_val_all),
            torch.FloatTensor(icr_val_all),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for _ in range(n_epochs):
        if staged:
            model.gate_network.train()
            model.change_expert.eval()
            model.icr_expert.eval()
        else:
            model.train()

        total_train_loss = 0.0
        total_train_rows = 0
        for batch_gate, batch_change, batch_icr, batch_y in train_loader:
            batch_gate = batch_gate.to(device)
            batch_change = batch_change.to(device)
            batch_icr = batch_icr.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = _forward_gated_probe(
                model,
                batch_gate,
                batch_change,
                batch_icr,
                experts_frozen=staged,
            )
            loss = criterion(outputs["probability"].view(-1), batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * len(batch_y)
            total_train_rows += len(batch_y)

        if val_loader is None:
            val_loss = total_train_loss / max(total_train_rows, 1)
        else:
            model.eval()
            total_val_loss = 0.0
            total_val_rows = 0
            with torch.no_grad():
                for batch_gate, batch_change, batch_icr, batch_y in val_loader:
                    batch_gate = batch_gate.to(device)
                    batch_change = batch_change.to(device)
                    batch_icr = batch_icr.to(device)
                    batch_y = batch_y.to(device)
                    outputs = _forward_gated_probe(
                        model,
                        batch_gate,
                        batch_change,
                        batch_icr,
                        experts_frozen=staged,
                    )
                    loss = criterion(outputs["probability"].view(-1), batch_y)
                    total_val_loss += float(loss.item()) * len(batch_y)
                    total_val_rows += len(batch_y)
            val_loss = total_val_loss / max(total_val_rows, 1)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = _clone_state_dict(model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError("Gated probe training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    probabilities: list[np.ndarray] = []
    gate_values: list[np.ndarray] = []
    change_probabilities: list[np.ndarray] = []
    icr_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch_gate, batch_change, batch_icr in predict_loader:
            batch_gate = batch_gate.to(device)
            batch_change = batch_change.to(device)
            batch_icr = batch_icr.to(device)
            outputs = model(batch_gate, batch_change, batch_icr)
            probabilities.append(outputs["probability"].view(-1).cpu().numpy())
            gate_values.append(outputs["gate"].view(-1).cpu().numpy())
            change_probabilities.append(outputs["change_probability"].view(-1).cpu().numpy())
            icr_probabilities.append(outputs["icr_probability"].view(-1).cpu().numpy())

    return FoldPrediction(
        probabilities=np.concatenate(probabilities).astype(np.float32),
        gate_values=np.concatenate(gate_values).astype(np.float32),
        change_probabilities=np.concatenate(change_probabilities).astype(np.float32),
        icr_probabilities=np.concatenate(icr_probabilities).astype(np.float32),
    )


def _run_baseline_variant(
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
) -> dict[str, Any]:
    folds = build_group_folds(sample_ids, sample_labels, n_splits=n_splits, seed=seed)
    ensure_parent_dir(output_dir / "placeholder")

    model_name = "baseline_mlp"
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

        scaler = ArrayScaler.fit(features[train_mask])
        train_x = scaler.transform(features[train_mask])
        val_x_all = scaler.transform(features[val_mask_all])
        val_x_labeled = scaler.transform(features[val_mask_labeled])

        fold_probs = _fit_baseline_mlp(
            x_train=train_x,
            y_train=silver_labels[train_mask].astype(np.float32),
            x_val_labeled=val_x_labeled,
            y_val_labeled=silver_labels[val_mask_labeled].astype(np.float32),
            x_val_all=val_x_all,
            device=device,
            epochs=TORCH_EPOCHS,
            batch_size=TORCH_BATCH_SIZE,
            learning_rate=TORCH_LEARNING_RATE,
            patience=TORCH_PATIENCE,
            weight_decay=TORCH_WEIGHT_DECAY,
            seed=seed + fold_id,
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
        "family": "cut_c_torch",
        "feature_set": feature_set_name,
        "model": model_name,
        "training_strategy": "supervised",
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
    print_metrics_summary(metrics_payload, prefix=f"{feature_set_name} / {model_name}")
    return metrics_payload


def _run_gated_variant(
    rows: Sequence[dict[str, Any]],
    feature_bundle: CutCFeatureBundle,
    silver_labels: np.ndarray,
    sample_ids: np.ndarray,
    sample_labels: np.ndarray,
    output_dir: Path,
    feature_set_name: str,
    n_splits: int,
    seed: int,
    top_k: int,
    device: str,
    staged: bool,
) -> dict[str, Any]:
    folds = build_group_folds(sample_ids, sample_labels, n_splits=n_splits, seed=seed)
    ensure_parent_dir(output_dir / "placeholder")

    model_name = "gated_probe"
    print(f"\n[{feature_set_name}] Training {model_name}")
    probabilities = np.full(len(rows), np.nan, dtype=np.float32)
    gate_values = np.full(len(rows), np.nan, dtype=np.float32)
    change_probabilities = np.full(len(rows), np.nan, dtype=np.float32)
    icr_probabilities = np.full(len(rows), np.nan, dtype=np.float32)
    fold_assignments = np.full(len(rows), -1, dtype=np.int32)
    span_metrics: list[dict[str, float]] = []
    sample_metrics = {"max": [], "topk_mean": [], "noisy_or": []}

    for fold_id, (train_samples, val_samples) in enumerate(folds):
        train_mask = np.isin(sample_ids, list(train_samples)) & (silver_labels >= 0)
        val_mask_all = np.isin(sample_ids, list(val_samples))
        val_mask_labeled = val_mask_all & (silver_labels >= 0)

        _check_binary_training_labels(silver_labels[train_mask], model_name=model_name, fold_id=fold_id)

        gate_scaler = ArrayScaler.fit(feature_bundle.gate_features[train_mask])
        change_scaler = ArrayScaler.fit(feature_bundle.change_features[train_mask])
        icr_scaler = ArrayScaler.fit(feature_bundle.icr_features[train_mask])

        fold_prediction = _fit_gated_probe(
            gate_train=gate_scaler.transform(feature_bundle.gate_features[train_mask]),
            change_train=change_scaler.transform(feature_bundle.change_features[train_mask]),
            icr_train=icr_scaler.transform(feature_bundle.icr_features[train_mask]),
            y_train=silver_labels[train_mask].astype(np.float32),
            gate_val_labeled=gate_scaler.transform(feature_bundle.gate_features[val_mask_labeled]),
            change_val_labeled=change_scaler.transform(feature_bundle.change_features[val_mask_labeled]),
            icr_val_labeled=icr_scaler.transform(feature_bundle.icr_features[val_mask_labeled]),
            y_val_labeled=silver_labels[val_mask_labeled].astype(np.float32),
            gate_val_all=gate_scaler.transform(feature_bundle.gate_features[val_mask_all]),
            change_val_all=change_scaler.transform(feature_bundle.change_features[val_mask_all]),
            icr_val_all=icr_scaler.transform(feature_bundle.icr_features[val_mask_all]),
            device=device,
            learning_rate=TORCH_LEARNING_RATE,
            batch_size=TORCH_BATCH_SIZE,
            patience=TORCH_PATIENCE,
            weight_decay=TORCH_WEIGHT_DECAY,
            joint_epochs=TORCH_EPOCHS,
            staged_expert_epochs=STAGED_EXPERT_EPOCHS,
            staged_gate_epochs=STAGED_GATE_EPOCHS,
            staged=staged,
            seed=seed + fold_id,
        )

        probabilities[val_mask_all] = fold_prediction.probabilities
        gate_values[val_mask_all] = fold_prediction.gate_values
        change_probabilities[val_mask_all] = fold_prediction.change_probabilities
        icr_probabilities[val_mask_all] = fold_prediction.icr_probabilities
        fold_assignments[val_mask_all] = fold_id

        if val_mask_labeled.any():
            span_metrics.append(
                evaluate_binary_predictions(
                    silver_labels[val_mask_labeled],
                    probabilities[val_mask_labeled],
                )
            )

        fold_rows = [rows[index] for index in np.where(val_mask_all)[0]]
        aggregated = aggregate_sample_predictions(fold_rows, fold_prediction.probabilities, top_k=top_k)
        for mode, payload in aggregated.items():
            sample_metrics[mode].append(evaluate_binary_predictions(payload["labels"], payload["probs"]))

    combined_feature_names = (
        [f"gate::{name}" for name in feature_bundle.gate_feature_names]
        + [f"change::{name}" for name in feature_bundle.change_feature_names]
        + [f"icr::{name}" for name in feature_bundle.icr_feature_names]
    )
    metrics_payload = {
        "family": "cut_c_torch",
        "feature_set": feature_set_name,
        "model": model_name,
        "training_strategy": "staged" if staged else "joint",
        "feature_names": combined_feature_names,
        "feature_dim": len(combined_feature_names),
        "feature_groups": {
            "gate": feature_bundle.gate_feature_names,
            "change_expert": feature_bundle.change_feature_names,
            "icr_expert": feature_bundle.icr_feature_names,
        },
        "feature_dims": {
            "gate": len(feature_bundle.gate_feature_names),
            "change_expert": len(feature_bundle.change_feature_names),
            "icr_expert": len(feature_bundle.icr_feature_names),
        },
        "gate_summary": _summarize_values(gate_values[~np.isnan(gate_values)]),
        "n_rows": len(rows),
        "n_labeled_rows": int((silver_labels >= 0).sum()),
        "span_level": summarize_metric_dicts(span_metrics),
        "sample_level": {mode: summarize_metric_dicts(metric_rows) for mode, metric_rows in sample_metrics.items()},
    }
    dump_json(output_dir / f"{model_name}.metrics.json", metrics_payload)
    write_jsonl(
        output_dir / f"{model_name}.oof_predictions.jsonl",
        _base_output_payload(
            rows,
            probabilities,
            fold_assignments,
            feature_set_name,
            model_name,
            extra_fields={
                "gate": gate_values,
                "change_probability": change_probabilities,
                "icr_probability": icr_probabilities,
            },
        ),
    )
    print_metrics_summary(metrics_payload, prefix=f"{feature_set_name} / {model_name}")
    return metrics_payload


def run_cut_c_experiment(combined_dataset_path: Path, output_dir: Path, device: str = "cpu") -> dict[str, Any]:
    """Run Cut C baselines and adaptive gated probes with grouped CV."""
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

    feature_bundle = build_cut_c_feature_bundle(icr_vectors=icr_vectors, entropy_vectors=entropy_vectors)
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    comparison_rows: list[dict[str, Any]] = []

    baseline_variants = [
        ("icr_only", feature_bundle.icr_features, feature_bundle.icr_feature_names),
        ("icr_entropy_concat", feature_bundle.concat_features, feature_bundle.concat_feature_names),
    ]
    for feature_set_name, features, feature_names in baseline_variants:
        metrics = _run_baseline_variant(
            rows=rows,
            features=features,
            feature_names=feature_names,
            silver_labels=silver_labels,
            sample_ids=sample_ids,
            sample_labels=sample_labels,
            output_dir=output_dir / feature_set_name / "torch",
            feature_set_name=feature_set_name,
            n_splits=5,
            seed=42,
            top_k=TOP_K,
            device=device,
        )
        all_results[feature_set_name] = {"torch": {metrics["model"]: metrics}}
        comparison_rows.append(
            {
                "feature_set": feature_set_name,
                "model": metrics["model"],
                "span_auroc": _metric_value(metrics, "span_level", "AUROC_mean"),
                "sample_auroc": _metric_value(metrics, "sample_level", "AUROC_mean", mode="max"),
                "sample_auprc": _metric_value(metrics, "sample_level", "AUPRC_mean", mode="max"),
                "sample_f1": _metric_value(metrics, "sample_level", "F1_mean", mode="max"),
            }
        )

    for feature_set_name, staged in (("gated_joint", False), ("gated_staged", True)):
        metrics = _run_gated_variant(
            rows=rows,
            feature_bundle=feature_bundle,
            silver_labels=silver_labels,
            sample_ids=sample_ids,
            sample_labels=sample_labels,
            output_dir=output_dir / feature_set_name / "torch",
            feature_set_name=feature_set_name,
            n_splits=5,
            seed=42,
            top_k=TOP_K,
            device=device,
            staged=staged,
        )
        all_results[feature_set_name] = {"torch": {metrics["model"]: metrics}}
        comparison_rows.append(
            {
                "feature_set": feature_set_name,
                "model": metrics["model"],
                "span_auroc": _metric_value(metrics, "span_level", "AUROC_mean"),
                "sample_auroc": _metric_value(metrics, "sample_level", "AUROC_mean", mode="max"),
                "sample_auprc": _metric_value(metrics, "sample_level", "AUPRC_mean", mode="max"),
                "sample_f1": _metric_value(metrics, "sample_level", "F1_mean", mode="max"),
            }
        )

    comparison_table = _build_comparison_table(comparison_rows)
    print("\nCut C comparison table")
    print(comparison_table)

    best_row = max(
        comparison_rows,
        key=lambda row: (
            float("-inf") if row["sample_auroc"] is None else float(row["sample_auroc"]),
            float("-inf") if row["span_auroc"] is None else float(row["span_auroc"]),
        ),
    )

    summary = {
        "combined_dataset_path": str(combined_dataset_path),
        "device": device,
        "results": all_results,
        "comparison_rows": comparison_rows,
        "comparison_table": comparison_table,
        "best_model": best_row,
    }
    dump_json(output_dir / "training_summary.json", summary)
    (output_dir / "comparison_table.txt").write_text(comparison_table + "\n", encoding="utf-8")
    return summary


__all__ = ["run_cut_c_experiment"]
