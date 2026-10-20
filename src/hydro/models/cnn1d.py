"""Multi-rate 1D CNN architecture and training pipeline for hydraulic diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS

logger = logging.getLogger(__name__)

TARGET_HEADS: tuple[str, ...] = ("cooler", "valve", "pump", "accumulator")
DEFAULT_HEAD_CLASSES: dict[str, int] = {
    "cooler": 3,
    "valve": 4,
    "pump": 3,
    "accumulator": 4,
}
DEVICE: torch.device = torch.device("cpu")


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Select compute device, defaulting to CUDA if available else CPU."""
    if device is not None:
        return torch.device(device)
    target = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(target)


resolve_device = get_device


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and configuration for 1D CNN training."""

    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    device: str | None = None


def _make_conv_block(
    in_ch: int,
    out_ch: int,
    spec: tuple[int, int, int],
) -> nn.Sequential:
    """Build a Conv1d-BatchNorm1d-ReLU-MaxPool1d block from (kernel, stride, pool) spec."""
    kernel, stride, pool = spec
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.MaxPool1d(pool),
    )


def _build_branches(
    n_ch_100: int,
    n_ch_10: int,
    n_ch_1: int,
) -> tuple[nn.Sequential, nn.Sequential, nn.Sequential]:
    """Build the three multi-rate feature extraction branches."""
    b100 = nn.Sequential(
        _make_conv_block(n_ch_100, 32, (7, 2, 4)),
        _make_conv_block(32, 64, (5, 2, 4)),
        _make_conv_block(64, 128, (3, 2, 2)),
        nn.AdaptiveAvgPool1d(1),
    )
    b10 = nn.Sequential(
        _make_conv_block(n_ch_10, 32, (5, 1, 2)),
        _make_conv_block(32, 64, (3, 1, 2)),
        _make_conv_block(64, 64, (3, 1, 2)),
        nn.AdaptiveAvgPool1d(1),
    )
    b1 = nn.Sequential(
        _make_conv_block(n_ch_1, 32, (3, 1, 2)),
        _make_conv_block(32, 64, (3, 1, 2)),
        nn.AdaptiveAvgPool1d(1),
    )
    return b100, b10, b1


class MultiRateCNN(nn.Module):
    """Multi-rate 1D convolutional network with branch fusion for condition monitoring."""

    def __init__(
        self,
        n_ch_100: int = 7,
        n_ch_10: int = 2,
        n_ch_1: int = 5,
        n_classes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        heads_cfg = n_classes or DEFAULT_HEAD_CLASSES
        self.branch_100, self.branch_10, self.branch_1 = _build_branches(n_ch_100, n_ch_10, n_ch_1)
        self.shared = nn.Linear(128 + 64 + 64, 128)
        self.heads = nn.ModuleDict({k: nn.Linear(128, v) for k, v in heads_cfg.items()})

    def forward(
        self,
        x100: torch.Tensor,
        x10: torch.Tensor,
        x1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward multi-rate input tensors through branches and target heads."""
        f100 = torch.flatten(self.branch_100(x100), start_dim=1)
        f10 = torch.flatten(self.branch_10(x10), start_dim=1)
        f1 = torch.flatten(self.branch_1(x1), start_dim=1)
        feat = torch.cat([f100, f10, f1], dim=1)
        h = torch.relu(self.shared(feat))
        return {name: head(h) for name, head in self.heads.items()}


def fit_channel_stats(
    data: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Compute per-channel mean and std on training rows only to avoid leakage."""
    stats: dict[str, tuple[float, float]] = {}
    idx_arr = np.atleast_1d(train_idx)
    for ch, arr in data.items():
        train_slice = arr[idx_arr]
        stats[ch] = (float(np.mean(train_slice)), float(np.std(train_slice)))
    return stats


def _normalize_rate_group(
    data: dict[str, np.ndarray],
    channels: list[str],
    idx: np.ndarray,
    stats: dict[str, tuple[float, float]],
) -> torch.Tensor:
    """Standardize and stack channels for a given rate group into a float32 tensor."""
    group_arrays: list[np.ndarray] = []
    idx_arr = np.atleast_1d(idx)
    for ch in channels:
        if ch not in data:
            raise ValueError(f"Channel {ch} missing from data")
        if ch not in stats:
            raise ValueError(f"Channel {ch} missing from channel stats")
        mean, std = stats[ch]
        denom = std if std > 1e-8 else 1.0
        arr = data[ch][idx_arr]
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        norm = (arr - mean) / denom
        group_arrays.append(norm)
    stacked = np.stack(group_arrays, axis=1)
    return torch.tensor(stacked, dtype=torch.float32)


def prepare_inputs(
    data: dict[str, np.ndarray],
    idx: np.ndarray,
    stats: dict[str, tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare standardized float32 input tensors grouped by rate (100, 10, 1 Hz)."""
    ch_100 = [c for c in PHYSICAL_CHANNELS if CHANNEL_REGISTRY[c].rate_hz == 100]
    ch_10 = [c for c in PHYSICAL_CHANNELS if CHANNEL_REGISTRY[c].rate_hz == 10]
    ch_1 = [c for c in PHYSICAL_CHANNELS if CHANNEL_REGISTRY[c].rate_hz == 1]
    t100 = _normalize_rate_group(data, ch_100, idx, stats)
    t10 = _normalize_rate_group(data, ch_10, idx, stats)
    t1 = _normalize_rate_group(data, ch_1, idx, stats)
    return t100, t10, t1


def encode_targets(
    targets: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Encode targets to 0-indexed integer class codes and sorted class values."""
    codes: dict[str, np.ndarray] = {}
    classes: dict[str, np.ndarray] = {}
    for col in TARGET_HEADS:
        if col not in targets.columns:
            raise ValueError(f"Target column {col} missing from targets dataframe")
        vals = targets[col].to_numpy()
        unique_vals = np.sort(np.unique(vals))
        codes[col] = np.searchsorted(unique_vals, vals)
        classes[col] = unique_vals
    return codes, classes


def _build_dataloader(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    codes: dict[str, np.ndarray],
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Build a seeded DataLoader from inputs and target codes."""
    y = [torch.tensor(codes[k], dtype=torch.long) for k in TARGET_HEADS]
    dataset = TensorDataset(*tensors, *y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def _train_epoch(
    model: MultiRateCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Run one training epoch and return average total loss across heads."""
    model.train()
    device = next(model.parameters()).device
    total_loss = 0.0
    for b_x100, b_x10, b_x1, b_c, b_v, b_p, b_a in loader:
        x = (b_x100.to(device), b_x10.to(device), b_x1.to(device))
        yb = [b_c.to(device), b_v.to(device), b_p.to(device), b_a.to(device)]
        optimizer.zero_grad()
        logits = model(*x)
        loss = sum(criterion(logits[k], yb[i]) for i, k in enumerate(TARGET_HEADS))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def _setup_training(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    train_idx: np.ndarray,
    config: TrainConfig,
) -> tuple[DataLoader, dict, dict[str, int]]:
    """Prepare dataloader, metadata, and head dimensions for training."""
    stats = fit_channel_stats(data, train_idx)
    codes, classes = encode_targets(targets.iloc[train_idx].reset_index(drop=True))
    tensors = prepare_inputs(data, train_idx, stats)
    loader = _build_dataloader(tensors, codes, config.batch_size, config.seed)
    n_classes = {k: len(classes[k]) for k in TARGET_HEADS}
    return loader, {"stats": stats, "classes": classes}, n_classes


def train_cnn_model(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    train_idx: np.ndarray,
    config: TrainConfig | None = None,
    **overrides: Any,
) -> tuple[MultiRateCNN, dict]:
    """Train MultiRateCNN with Adam optimizer and sum of cross-entropy losses."""
    cfg = config or TrainConfig()
    if overrides:
        cfg = replace(cfg, **overrides)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(cfg.seed)  # noqa: NPY002
    loader, meta, n_classes = _setup_training(data, targets, train_idx, cfg)
    device = get_device(cfg.device)
    logger.info("Training MultiRateCNN on %s", device)
    model = MultiRateCNN(n_classes=n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    meta["loss_history"] = [
        _train_epoch(model, loader, optimizer, criterion) for _ in range(cfg.epochs)
    ]
    model.to("cpu")
    model.eval()
    return model, meta


def predict_cnn(
    model: MultiRateCNN,
    data: dict[str, np.ndarray],
    idx: np.ndarray,
    meta: dict,
) -> pd.DataFrame:
    """Predict decoded condition classes for hydraulic targets using trained CNN."""
    model.eval()
    stats = meta["stats"]
    classes = meta["classes"]
    x100, x10, x1 = prepare_inputs(data, idx, stats)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(x100.to(device), x10.to(device), x1.to(device))
    preds: dict[str, np.ndarray] = {}
    for col in TARGET_HEADS:
        code_preds = torch.argmax(logits[col], dim=-1).cpu().numpy()
        preds[col] = classes[col][code_preds]
    return pd.DataFrame(preds)
