"""Adaptive gating modules and feature preparation for Cut C."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_ROOT = SCRIPT_DIR.parent
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

from shared.paths import ensure_spanlab_importable


ensure_spanlab_importable()

from spanlab.dependencies import require_torch
from spanlab.features import extract_change_point_features


try:
    torch = require_torch()
    nn = torch.nn
    F = torch.nn.functional
except RuntimeError:
    torch = None
    nn = None
    F = None


_ModuleBase = nn.Module if nn is not None else object


GATE_ENTROPY_LAYERS = slice(13, 18)


def _validate_icr_vectors(icr_vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(icr_vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected ICR vectors with shape [N, n_layers], got {matrix.shape}.")
    if matrix.shape[1] < 27:
        raise ValueError(f"Expected at least 27 ICR layers, got {matrix.shape[1]}.")
    return matrix[:, :27].astype(np.float32)


def _validate_entropy_vectors(entropy_vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(entropy_vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected entropy vectors with shape [N, n_layers], got {matrix.shape}.")
    if matrix.shape[1] < GATE_ENTROPY_LAYERS.stop:
        raise ValueError(
            f"Expected at least {GATE_ENTROPY_LAYERS.stop} entropy layers, got {matrix.shape[1]}."
        )
    return matrix.astype(np.float32)


@dataclass(frozen=True)
class ArrayScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "ArrayScaler":
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected feature matrix with shape [N, D], got {matrix.shape}.")
        mean = matrix.mean(axis=0, keepdims=True).astype(np.float32)
        std = matrix.std(axis=0, keepdims=True).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float32)
        return ((matrix - self.mean) / self.std).astype(np.float32)


@dataclass(frozen=True)
class CutCFeatureBundle:
    gate_features: np.ndarray
    change_features: np.ndarray
    icr_features: np.ndarray
    concat_features: np.ndarray
    gate_feature_names: list[str]
    change_feature_names: list[str]
    icr_feature_names: list[str]
    concat_feature_names: list[str]


def _require_torch_runtime() -> tuple[object, object, object]:
    global torch, nn, F
    if torch is None or nn is None or F is None:
        torch = require_torch()
        nn = torch.nn
        F = torch.nn.functional
    return torch, nn, F


class BinaryMLP(_ModuleBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        _require_torch_runtime()
        super().__init__()
        hidden_1, hidden_2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_2, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        out = self.dropout1(out)
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        out = self.dropout2(out)
        return torch.sigmoid(self.fc3(out))


class GateMLP(_ModuleBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (16, 8),
        dropout: float = 0.1,
    ) -> None:
        _require_torch_runtime()
        super().__init__()
        hidden_1, hidden_2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_2, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        out = self.dropout1(out)
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        out = self.dropout2(out)
        return torch.sigmoid(self.fc3(out))


class GatedProbe(_ModuleBase):
    """Mixture-of-experts probe that routes between raw ICR and change-point experts."""

    def __init__(
        self,
        gate_input_dim: int,
        change_input_dim: int,
        icr_input_dim: int,
    ) -> None:
        _require_torch_runtime()
        super().__init__()
        self.gate_network = GateMLP(input_dim=gate_input_dim)
        self.change_expert = BinaryMLP(input_dim=change_input_dim, hidden_dims=(32, 16), dropout=0.2)
        self.icr_expert = BinaryMLP(input_dim=icr_input_dim, hidden_dims=(64, 32), dropout=0.3)

    def freeze_experts(self) -> None:
        for expert in (self.change_expert, self.icr_expert):
            expert.eval()
            for parameter in expert.parameters():
                parameter.requires_grad = False

    def unfreeze_all(self) -> None:
        for module in (self.gate_network, self.change_expert, self.icr_expert):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        gate_x: torch.Tensor,
        change_x: torch.Tensor,
        icr_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gate = self.gate_network(gate_x)
        change_probability = self.change_expert(change_x)
        icr_probability = self.icr_expert(icr_x)
        probability = gate * change_probability + (1.0 - gate) * icr_probability
        return {
            "probability": probability,
            "gate": gate,
            "change_probability": change_probability,
            "icr_probability": icr_probability,
        }


def build_cut_c_feature_bundle(icr_vectors: np.ndarray, entropy_vectors: np.ndarray) -> CutCFeatureBundle:
    """Prepare the gate and expert inputs used by Cut C."""
    icr_matrix = _validate_icr_vectors(icr_vectors)
    entropy_matrix = _validate_entropy_vectors(entropy_vectors)

    gate_features = entropy_matrix[:, GATE_ENTROPY_LAYERS].astype(np.float32)
    change_features, change_feature_names = extract_change_point_features(icr_matrix)
    change_features = np.asarray(change_features, dtype=np.float32)

    icr_feature_names = [f"icr_layer_{index:02d}" for index in range(icr_matrix.shape[1])]
    gate_feature_names = [f"entropy_layer_{index:02d}" for index in range(GATE_ENTROPY_LAYERS.start, GATE_ENTROPY_LAYERS.stop)]
    concat_features = np.concatenate([icr_matrix, gate_features], axis=1).astype(np.float32)
    concat_feature_names = icr_feature_names + gate_feature_names

    return CutCFeatureBundle(
        gate_features=gate_features,
        change_features=change_features,
        icr_features=icr_matrix,
        concat_features=concat_features,
        gate_feature_names=gate_feature_names,
        change_feature_names=list(change_feature_names),
        icr_feature_names=icr_feature_names,
        concat_feature_names=concat_feature_names,
    )


__all__ = [
    "ArrayScaler",
    "BinaryMLP",
    "CutCFeatureBundle",
    "GATE_ENTROPY_LAYERS",
    "GateMLP",
    "GatedProbe",
    "build_cut_c_feature_bundle",
]
