"""Centralized MLP training (single-node, all training data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.mlp import (
    IDSMLP, NumpyDataset, create_model, evaluate_model,
    load_checkpoint, save_checkpoint, save_model, train_epoch,
)


class CentralizedTrainer:
    """Centralized MLP training on the full training split."""

    def __init__(self, train_X: np.ndarray, train_y: np.ndarray,
                 test_X: np.ndarray, test_y: np.ndarray,
                 num_classes: int = 10, input_dim: int = 41,
                 seed: int = 42, device: str | torch.device = "cpu",
                 checkpoint_dir: str | Path = "results/m3/checkpoints"):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.seed = seed
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = create_model(input_dim=input_dim,
                                  num_classes=num_classes,
                                  seed=seed).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.history: list[dict[str, Any]] = []

    def train(self, epochs: int = 30, batch_size: int = 256,
              lr: float = 0.001,
              resume_from: str | Path | None = None) -> dict[str, Any]:
        """Train centralized MLP on the full training split.

        Epochs reduced to 30 for CPU runtime (was 90 = 30 FedAvg rounds x 3
        local epochs). Architecture and hyperparameters unchanged.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_ds = NumpyDataset(self.train_X, self.train_y)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,
                                  generator=torch.Generator().manual_seed(self.seed))
        test_loader = DataLoader(
            NumpyDataset(self.test_X, self.test_y),
            batch_size=batch_size, shuffle=False)

        start_epoch = 0
        if resume_from is not None:
            ckpt_path = Path(resume_from)
            if ckpt_path.exists():
                extra = load_checkpoint(self.model, optimizer, ckpt_path)
                start_epoch = extra.get("round", 0) + 1
                history_path = self.checkpoint_dir / "history.json"
                if history_path.exists():
                    self.history = json.loads(history_path.read_text())
                print(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            train_metrics = train_epoch(self.model, train_loader, optimizer,
                                        self.criterion, self.device)
            test_metrics = evaluate_model(self.model, test_loader,
                                          self.criterion, self.device)

            record = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
            }
            self.history.append(record)

            ckpt_path = self.checkpoint_dir / f"centralized_epoch_{epoch:03d}.pt"
            save_checkpoint(self.model, optimizer, epoch, ckpt_path,
                            extra={"history": self.history})
            (self.checkpoint_dir / "history.json").write_text(
                json.dumps(self.history, indent=2))

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"train_acc={train_metrics['accuracy']:.4f}, "
                      f"test_acc={test_metrics['accuracy']:.4f}")

        # Final evaluation with predictions
        final_metrics = evaluate_model(self.model, test_loader,
                                       self.criterion, self.device)
        save_model(self.model, self.checkpoint_dir / "final_model.pt")
        return {
            "history": self.history,
            "final_metrics": {
                "loss": float(final_metrics["loss"]),
                "accuracy": float(final_metrics["accuracy"]),
            },
            "test_predictions": final_metrics.pop("predictions").tolist(),
            "test_targets": final_metrics.pop("targets").tolist(),
            "test_probabilities": final_metrics.pop("probabilities").tolist(),
            "final_model_path": str(self.checkpoint_dir / "final_model.pt"),
        }
