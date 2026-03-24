import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .aggregation import aggregate_probabilities
from .dependencies import require_matplotlib
from .io_utils import ensure_parent_dir, read_jsonl


ROUTE_COLORS = {
    "tokenizer_windows_dataset": "#c56a1a",
    "spacy_spans_dataset": "#1f8a70",
}


def prettify_dataset_name(dataset_name: str) -> str:
    if dataset_name == "tokenizer_windows_dataset":
        return "Tokenizer Window"
    if dataset_name == "spacy_spans_dataset":
        return "spaCy Span"
    return dataset_name.replace("_", " ")


def prettify_model_name(name: str) -> str:
    mapping = {
        "BaselineMLP": "Baseline MLP",
        "LogisticRegression": "LR",
        "RandomForest": "RF",
        "TemporalCNN": "Temporal CNN",
        "MultiScaleCNN": "MultiScale CNN",
        "GRUEncoder": "GRU",
        "SmallTransformer": "Transformer",
        "Deep1DCNN": "Deep1DCNN",
    }
    return mapping.get(name, name)


def load_metric_records(results_root: Path) -> List[Dict]:
    records: List[Dict] = []
    for metrics_path in sorted(results_root.rglob("*.metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset_name = metrics_path.parent.parent.name
        family_name = metrics_path.parent.name
        records.append(
            {
                "metrics_path": str(metrics_path),
                "dataset_name": dataset_name,
                "family_name": family_name,
                "model_name": payload.get("model", metrics_path.stem),
                "payload": payload,
            }
        )
    return records


def plot_method_summary(records: Sequence[Dict], output_path: Path) -> None:
    if not records:
        raise ValueError("No metric records found for plotting.")

    matplotlib = require_matplotlib()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{prettify_dataset_name(record['dataset_name'])}\n{prettify_model_name(record['model_name'])}"
        for record in records
    ]
    colors = [ROUTE_COLORS.get(record["dataset_name"], "#4c78a8") for record in records]
    span_aurocs = [record["payload"]["span_level"].get("AUROC_mean", 0.0) for record in records]
    sample_aurocs = [
        record["payload"]["sample_level"].get("noisy_or", {}).get("AUROC_mean", 0.0)
        for record in records
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(records) * 1.8), 5.5))
    for ax, values, title in [
        (axes[0], span_aurocs, "Span-Level AUROC"),
        (axes[1], sample_aurocs, "Sample-Level AUROC (Noisy-Or)"),
    ]:
        bars = ax.bar(np.arange(len(labels)), values, color=colors, alpha=0.9)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(True, alpha=0.25, axis="y")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Span Lab Method Summary", fontsize=14)
    plt.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aggregation_summary(records: Sequence[Dict], output_path: Path) -> None:
    if not records:
        raise ValueError("No metric records found for plotting.")

    matplotlib = require_matplotlib()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["max", "topk_mean", "noisy_or"]
    fig, ax = plt.subplots(figsize=(max(10, len(records) * 2.0), 5.5))
    x = np.arange(len(records))
    width = 0.22
    colors = {"max": "#8c564b", "topk_mean": "#2ca02c", "noisy_or": "#1f77b4"}

    for offset, mode in enumerate(modes):
        values = [record["payload"]["sample_level"].get(mode, {}).get("AUROC_mean", 0.0) for record in records]
        bars = ax.bar(x + (offset - 1) * width, values, width=width, label=mode, color=colors[mode], alpha=0.9)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    labels = [
        f"{prettify_dataset_name(record['dataset_name'])}\n{prettify_model_name(record['model_name'])}"
        for record in records
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Sample Aggregation Comparison")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()

    plt.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_dataset_by_span_length(
    dataset_rows: Sequence[Dict],
    prediction_by_span: Optional[Dict[str, float]] = None,
) -> Dict:
    count_by_len = Counter()
    labeled_by_len = Counter()
    positive_by_len = Counter()
    score_sum_by_len = Counter()
    score_count_by_len = Counter()
    window_size_counts = Counter()
    span_type_counts = Counter()

    for row in dataset_rows:
        span_len = int(row["span_len_tokens"])
        count_by_len[span_len] += 1
        span_type_counts[row["span_type"]] += 1
        if "window_size" in row:
            window_size_counts[int(row["window_size"])] += 1
        if row.get("silver_label") is not None:
            labeled_by_len[span_len] += 1
            positive_by_len[span_len] += int(row["silver_label"])
        if prediction_by_span is not None and row["span_id"] in prediction_by_span:
            score_sum_by_len[span_len] += float(prediction_by_span[row["span_id"]])
            score_count_by_len[span_len] += 1

    span_lengths = sorted(count_by_len)
    positive_rate = [
        positive_by_len[length] / max(labeled_by_len[length], 1)
        for length in span_lengths
    ]
    mean_score = [
        score_sum_by_len[length] / max(score_count_by_len[length], 1)
        if score_count_by_len[length] > 0
        else np.nan
        for length in span_lengths
    ]

    return {
        "span_lengths": span_lengths,
        "count_by_len": [count_by_len[length] for length in span_lengths],
        "positive_rate_by_len": positive_rate,
        "mean_score_by_len": mean_score,
        "window_size_counts": dict(sorted(window_size_counts.items())),
        "span_type_counts": dict(sorted(span_type_counts.items())),
    }


def plot_span_length_statistics(
    dataset_rows: Sequence[Dict],
    output_path: Path,
    prediction_by_span: Optional[Dict[str, float]] = None,
) -> None:
    stats = summarize_dataset_by_span_length(dataset_rows, prediction_by_span=prediction_by_span)

    matplotlib = require_matplotlib()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    axes[0].bar(stats["span_lengths"], stats["count_by_len"], color="#4c78a8")
    axes[0].set_title("Span Count by Token Length")
    axes[0].set_xlabel("Span Length (tokens)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.25, axis="y")

    axes[1].plot(stats["span_lengths"], stats["positive_rate_by_len"], marker="o", color="#d62728")
    axes[1].set_title("Silver Positive Rate by Span Length")
    axes[1].set_xlabel("Span Length (tokens)")
    axes[1].set_ylabel("Positive Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(stats["span_lengths"], stats["mean_score_by_len"], marker="o", color="#2ca02c")
    axes[2].set_title("Mean Predicted Probability by Span Length")
    axes[2].set_xlabel("Span Length (tokens)")
    axes[2].set_ylabel("Mean Probability")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.25)

    if stats["window_size_counts"]:
        labels = list(stats["window_size_counts"].keys())
        values = list(stats["window_size_counts"].values())
        axes[3].bar(labels, values, color="#9467bd")
        axes[3].set_title("Tokenizer Window Size Counts")
        axes[3].set_xlabel("Window Size")
        axes[3].set_ylabel("Count")
    else:
        labels = list(stats["span_type_counts"].keys())
        values = list(stats["span_type_counts"].values())
        axes[3].bar(labels, values, color="#9467bd")
        axes[3].set_title("Span Type Counts")
        axes[3].set_xlabel("Span Type")
        axes[3].set_ylabel("Count")
    axes[3].grid(True, alpha=0.25, axis="y")

    fig.suptitle("Span Statistics", fontsize=14)
    plt.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_prediction_index(prediction_rows: Sequence[Dict]) -> Dict[str, float]:
    return {
        row["span_id"]: float(row["probability"])
        for row in prediction_rows
        if row.get("probability") is not None
    }


def aggregate_sample_scores(prediction_rows: Sequence[Dict], mode: str = "noisy_or") -> Dict[str, Dict]:
    grouped_probs: Dict[str, List[float]] = defaultdict(list)
    sample_labels: Dict[str, int] = {}
    for row in prediction_rows:
        if row.get("probability") is None:
            continue
        grouped_probs[row["sample_id"]].append(float(row["probability"]))
        sample_labels[row["sample_id"]] = int(row["sample_label"])
    aggregated = {}
    for sample_id, probs in grouped_probs.items():
        aggregated[sample_id] = {
            "score": aggregate_probabilities(probs, mode=mode),
            "label": sample_labels[sample_id],
        }
    return aggregated


def select_case_sample_id(
    prediction_rows: Sequence[Dict],
    selection: str = "highest_hallucinated",
    aggregation_mode: str = "noisy_or",
) -> str:
    sample_scores = aggregate_sample_scores(prediction_rows, mode=aggregation_mode)
    candidates = list(sample_scores.items())
    if not candidates:
        raise ValueError("No prediction rows with probabilities available.")

    if selection == "highest_hallucinated":
        candidates = [(sample_id, payload) for sample_id, payload in candidates if payload["label"] == 1]
        key_fn = lambda item: item[1]["score"]
        reverse = True
    elif selection == "highest_false_positive":
        candidates = [(sample_id, payload) for sample_id, payload in candidates if payload["label"] == 0]
        key_fn = lambda item: item[1]["score"]
        reverse = True
    elif selection == "lowest_hallucinated":
        candidates = [(sample_id, payload) for sample_id, payload in candidates if payload["label"] == 1]
        key_fn = lambda item: item[1]["score"]
        reverse = False
    else:
        raise ValueError(f"Unsupported selection mode: {selection}")

    if not candidates:
        raise ValueError(f"No candidates found for selection mode: {selection}")
    return sorted(candidates, key=key_fn, reverse=reverse)[0][0]


def build_token_level_scores(
    dataset_rows: Sequence[Dict],
    prediction_by_span: Dict[str, float],
    n_tokens: int,
) -> Tuple[np.ndarray, np.ndarray]:
    token_scores = np.zeros(n_tokens, dtype=np.float32)
    token_silver = np.zeros(n_tokens, dtype=np.int32)

    for row in dataset_rows:
        probability = prediction_by_span.get(row["span_id"])
        if probability is None:
            continue
        token_start = int(row["token_start"])
        token_end = int(row["token_end"])
        token_scores[token_start:token_end] = np.maximum(token_scores[token_start:token_end], probability)

        silver_label = row.get("silver_label")
        if silver_label is None:
            continue
        if int(silver_label) == 1:
            token_silver[token_start:token_end] = 1
        else:
            token_silver[token_start:token_end] = np.where(
                token_silver[token_start:token_end] == 1,
                1,
                -1,
            )

    return token_scores, token_silver


def _format_token_label(text: str, max_len: int = 10) -> str:
    cleaned = text.replace("\n", "\\n")
    if cleaned.strip() == "":
        cleaned = "<sp>"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def plot_case_heatmap(
    sample_row: Dict,
    dataset_rows: Sequence[Dict],
    prediction_rows: Sequence[Dict],
    output_path: Path,
    aggregation_mode: str = "noisy_or",
    top_n_spans: int = 8,
) -> None:
    matplotlib = require_matplotlib()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prediction_by_span = build_prediction_index(prediction_rows)
    n_tokens = len(sample_row["response_token_ids"])
    token_scores, token_silver = build_token_level_scores(dataset_rows, prediction_by_span, n_tokens=n_tokens)
    sample_score = aggregate_probabilities(
        [float(row["probability"]) for row in prediction_rows if row.get("probability") is not None],
        mode=aggregation_mode,
    )
    icr_matrix = np.asarray(sample_row["icr_scores"], dtype=np.float32)

    token_labels = [_format_token_label(text) for text in sample_row["response_token_texts"]]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.6, 1.7], height_ratios=[4.0, 1.3], wspace=0.25, hspace=0.2)
    heat_ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[:, 1])
    bar_ax = fig.add_subplot(gs[1, 0])

    im = heat_ax.imshow(icr_matrix, aspect="auto", cmap="viridis")
    heat_ax.set_title(
        f"ICR Heatmap | sample_id={sample_row['sample_id']} | sample_label={sample_row['sample_label']} | "
        f"agg={sample_score:.3f}"
    )
    heat_ax.set_ylabel("Layer")
    heat_ax.set_xticks(np.arange(n_tokens))
    heat_ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    heat_ax.set_xlabel("Response Tokens")
    fig.colorbar(im, ax=heat_ax, fraction=0.03, pad=0.02)

    bar_colors = []
    for label in token_silver:
        if label == 1:
            bar_colors.append("#d62728")
        elif label == -1:
            bar_colors.append("#1f77b4")
        else:
            bar_colors.append("#7f7f7f")
    bar_ax.bar(np.arange(n_tokens), token_scores, color=bar_colors, alpha=0.9)
    bar_ax.set_ylim(0.0, 1.0)
    bar_ax.set_ylabel("Token Risk")
    bar_ax.set_xlabel("Response Tokens")
    bar_ax.set_xticks(np.arange(n_tokens))
    bar_ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    bar_ax.grid(True, alpha=0.25, axis="y")

    ranked_spans = sorted(
        (
            (
                prediction_by_span.get(row["span_id"], float("-inf")),
                row,
            )
            for row in dataset_rows
            if row["span_id"] in prediction_by_span
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:top_n_spans]

    info_ax.axis("off")
    wrapped_response = "\n".join(textwrap.wrap(sample_row["response"], width=34))
    info_lines = [
        "Response",
        wrapped_response,
        "",
        f"Question: {sample_row.get('question', '')[:80]}",
        "",
        "Top spans",
    ]
    for probability, row in ranked_spans:
        silver_label = row.get("silver_label")
        info_lines.append(
            f"{probability:.3f} | [{row['token_start']}:{row['token_end']}] | "
            f"silver={silver_label} | {row['span_text'][:42]}"
        )
    info_ax.text(
        0.0,
        1.0,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
