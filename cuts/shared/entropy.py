"""Per-layer logit entropy computation from hidden states."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_layer_entropy(
    hidden_state: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute token-wise output entropy from a layer's hidden states.

    Args:
        hidden_state: ``[seq_len, hidden_dim]`` or ``[1, seq_len, hidden_dim]`` tensor.
        lm_head_weight: ``[vocab_size, hidden_dim]`` unembedding matrix.
        lm_head_bias: Optional ``[vocab_size]`` bias term.

    Returns:
        ``[seq_len]`` tensor of entropy values in nats.
    """
    if hidden_state.dim() == 3:
        hidden_state = hidden_state.squeeze(0)
    if hidden_state.dim() != 2:
        raise ValueError(f"Expected hidden_state to be 2D after squeeze, got shape {tuple(hidden_state.shape)}.")

    logits = hidden_state.float() @ lm_head_weight.float().T
    if lm_head_bias is not None:
        logits = logits + lm_head_bias.float()

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def compute_all_layer_entropies(
    hidden_states: tuple[torch.Tensor, ...],
    lm_head_weight: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor] = None,
    response_start: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """Compute per-layer entropy for response tokens only.

    Args:
        hidden_states: Tuple of hidden states from the model forward pass.
            Element 0 is the embedding layer and is skipped.
        lm_head_weight: Unembedding matrix.
        lm_head_bias: Optional unembedding bias.
        response_start: Index of the first response token in the sequence.
        device: Device used for the layer-wise projection.

    Returns:
        ``[n_layers, n_response_tokens]`` array of entropy values.
    """
    if len(hidden_states) < 2:
        raise ValueError("Expected hidden_states to include embedding + at least one transformer layer.")

    first_hidden = hidden_states[0]
    seq_len = first_hidden.shape[1] if first_hidden.dim() == 3 else first_hidden.shape[0]
    if response_start < 0 or response_start > seq_len:
        raise ValueError(f"response_start={response_start} is out of bounds for sequence length {seq_len}.")

    n_layers = len(hidden_states) - 1
    n_response_tokens = seq_len - response_start
    entropy_matrix = np.zeros((n_layers, n_response_tokens), dtype=np.float32)

    if n_response_tokens == 0:
        return entropy_matrix

    lm_head_weight = lm_head_weight.to(device)
    if lm_head_bias is not None:
        lm_head_bias = lm_head_bias.to(device)

    for layer_idx in range(n_layers):
        hidden = hidden_states[layer_idx + 1]
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)

        hidden_response = hidden[response_start:].to(device)
        with torch.no_grad():
            layer_entropy = compute_layer_entropy(hidden_response, lm_head_weight, lm_head_bias)

        entropy_matrix[layer_idx] = layer_entropy.detach().cpu().numpy()

        del hidden_response
        del layer_entropy

    return entropy_matrix


def delta_entropy(entropy_matrix: np.ndarray) -> np.ndarray:
    """Compute layer-to-layer entropy deltas."""
    matrix = np.asarray(entropy_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D entropy matrix, got shape {matrix.shape}.")
    return np.diff(matrix, axis=0)


def entropy_summary_stats(entropy_matrix: np.ndarray) -> dict[str, object]:
    """Return lightweight summary statistics for entropy sanity checks."""
    matrix = np.asarray(entropy_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D entropy matrix, got shape {matrix.shape}.")
    if matrix.size == 0:
        return {
            "shape": list(matrix.shape),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "shape": list(matrix.shape),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "mean": float(matrix.mean()),
        "std": float(matrix.std()),
    }


__all__ = [
    "compute_layer_entropy",
    "compute_all_layer_entropies",
    "delta_entropy",
    "entropy_summary_stats",
]
