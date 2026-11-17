"""MLP model for federated multiclass intrusion detection.

Architecture: 41 -> 128 (ReLU, Dropout 0.2) -> 64 (ReLU, Dropout 0.2) -> 10
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class IDSMLP(nn.Module):
    """Lightweight MLP for multiclass IDS."""

    def __init__(self, input_dim: int = 41, hidden1: int = 128, hidden2: int = 64,
                 num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_weights(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        state = {k: torch.tensor(v) for k, v in weights.items()}
        self.load_state_dict(state)


def create_model(input_dim: int = 41, num_classes: int = 10,
                 seed: int | None = None) -> IDSMLP:
    if seed is not None:
        torch.manual_seed(seed)
    return IDSMLP(input_dim=input_dim, num_classes=num_classes)


def train_epoch(model: IDSMLP, loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def evaluate_model(model: IDSMLP, loader: torch.utils.data.DataLoader,
                   criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": np.concatenate(all_preds),
        "targets": np.concatenate(all_targets),
        "probabilities": np.concatenate(all_probs),
    }


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def save_checkpoint(model: IDSMLP, optimizer: torch.optim.Optimizer,
                    round_num: int, path: Path,
                    extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "round": round_num,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(model: IDSMLP, optimizer: torch.optim.Optimizer,
                    path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return {k: v for k, v in ckpt.items() if k not in ("model_state", "optimizer_state")}


def average_weights(weight_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """FedAvg: simple average of model weights."""
    avg: dict[str, np.ndarray] = {}
    keys = weight_list[0].keys()
    for k in keys:
        avg[k] = np.mean([w[k] for w in weight_list], axis=0)
    return avg


def save_model(model: IDSMLP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, input_dim: int = 41, num_classes: int = 10,
               seed: int | None = None) -> IDSMLP:
    model = create_model(input_dim=input_dim, num_classes=num_classes, seed=seed)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
