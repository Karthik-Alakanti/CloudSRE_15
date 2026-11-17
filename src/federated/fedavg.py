"""Federated Averaging (FedAvg) training loop with checkpointing and resumability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.mlp import (
    IDSMLP, NumpyDataset, average_weights, create_model, evaluate_model,
    load_checkpoint, save_checkpoint, save_model, train_epoch,
)


class FedAvgTrainer:
    """Federated Averaging trainer for multiclass IDS."""

    def __init__(self, client_data: dict[int, dict[str, np.ndarray]],
                 test_X: np.ndarray, test_y: np.ndarray,
                 num_clients: int = 10, num_classes: int = 10,
                 input_dim: int = 41, seed: int = 42,
                 device: str | torch.device = "cpu",
                 checkpoint_dir: str | Path = "results/m3/checkpoints"):
        self.client_data = client_data
        self.test_X = test_X
        self.test_y = test_y
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.seed = seed
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_model = create_model(input_dim=input_dim,
                                         num_classes=num_classes,
                                         seed=seed).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.history: list[dict[str, Any]] = []
        self.start_round = 0

    def _client_dataloader(self, X: np.ndarray, y: np.ndarray,
                           batch_size: int, shuffle: bool,
                           seed: int) -> DataLoader:
        ds = NumpyDataset(X, y)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          generator=generator)

    def _local_train(self, client_id: int, global_weights: dict[str, np.ndarray],
                     local_epochs: int, batch_size: int, lr: float,
                     round_seed: int) -> dict[str, np.ndarray]:
        """Train a local model for one client and return updated weights."""
        train_X = self.client_data[client_id]["train_X"]
        train_y = self.client_data[client_id]["train_y"]

        local_model = create_model(input_dim=self.input_dim,
                                   num_classes=self.num_classes,
                                   seed=round_seed).to(self.device)
        local_model.load_state_dict({k: torch.tensor(v)
                                     for k, v in global_weights.items()})

        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        loader = self._client_dataloader(train_X, train_y, batch_size,
                                         shuffle=True, seed=round_seed)

        for _ in range(local_epochs):
            train_epoch(local_model, loader, optimizer, self.criterion, self.device)

        return local_model.get_weights()

    def _fedavg_round(self, round_num: int, local_epochs: int, batch_size: int,
                      lr: float, client_fraction: float = 1.0) -> dict[str, Any]:
        """Execute one FedAvg communication round."""
        round_seed = self.seed + round_num
        rng = np.random.default_rng(round_seed)

        # Select participating clients
        if client_fraction < 1.0:
            num_participants = max(1, int(self.num_clients * client_fraction))
            participating = sorted(rng.choice(self.num_clients,
                                              size=num_participants,
                                              replace=False).tolist())
        else:
            participating = list(range(self.num_clients))

        # Local training
        client_weights = []
        client_sizes = []
        for cid in participating:
            w = self._local_train(cid, self.global_model.get_weights(),
                                  local_epochs, batch_size, lr,
                                  round_seed + cid)
            client_weights.append(w)
            client_sizes.append(len(self.client_data[cid]["train_y"]))

        # Weighted average (by client training set size)
        total_size = sum(client_sizes)
        averaged: dict[str, np.ndarray] = {}
        keys = client_weights[0].keys()
        for k in keys:
            averaged[k] = sum(w[k] * (sz / total_size)
                              for w, sz in zip(client_weights, client_sizes))

        self.global_model.load_state_dict(
            {k: torch.tensor(v) for k, v in averaged.items()})

        # Evaluate on test set (aggregate metrics only — no per-sample
        # predictions stored per round to keep checkpoints small)
        test_loader = DataLoader(
            NumpyDataset(self.test_X, self.test_y),
            batch_size=batch_size, shuffle=False)
        test_metrics = evaluate_model(self.global_model, test_loader,
                                      self.criterion, self.device)

        round_record = {
            "round": round_num,
            "participating_clients": participating,
            "client_sizes": {str(c): s for c, s in zip(participating, client_sizes)},
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
        }
        return round_record

    def train(self, num_rounds: int = 30, local_epochs: int = 3,
              batch_size: int = 256, lr: float = 0.001,
              resume_from: str | Path | None = None) -> dict[str, Any]:
        """Run FedAvg training. Resumable from checkpoint."""
        if resume_from is not None:
            ckpt_path = Path(resume_from)
            if ckpt_path.exists():
                extra = load_checkpoint(self.global_model,
                                        torch.optim.Adam(self.global_model.parameters()),
                                        ckpt_path)
                self.start_round = extra.get("round", 0) + 1
                history_path = self.checkpoint_dir / "history.json"
                if history_path.exists():
                    self.history = json.loads(history_path.read_text())
                print(f"Resumed from round {self.start_round}")

        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr)

        for r in range(self.start_round, num_rounds):
            round_record = self._fedavg_round(
                r, local_epochs, batch_size, lr)
            self.history.append(round_record)

            # Save checkpoint every round
            ckpt_path = self.checkpoint_dir / f"fedavg_round_{r:03d}.pt"
            save_checkpoint(self.global_model, optimizer, r, ckpt_path,
                            extra={"history": self.history})

            # Save history
            (self.checkpoint_dir / "history.json").write_text(
                json.dumps(self.history, indent=2))

            if (r + 1) % 5 == 0 or r == num_rounds - 1:
                print(f"Round {r+1}/{num_rounds}: "
                      f"test_acc={round_record['test_accuracy']:.4f}, "
                      f"test_loss={round_record['test_loss']:.4f}")

        # Save final model
        save_model(self.global_model, self.checkpoint_dir / "final_model.pt")

        return {
            "history": self.history,
            "final_model_path": str(self.checkpoint_dir / "final_model.pt"),
        }

    def evaluate_final(self, batch_size: int = 256) -> dict[str, Any]:
        """Evaluate the final global model on test set."""
        test_loader = DataLoader(
            NumpyDataset(self.test_X, self.test_y),
            batch_size=batch_size, shuffle=False)
        metrics = evaluate_model(self.global_model, test_loader,
                                 self.criterion, self.device)
        return metrics
