"""
fungal_classifier/models/deep_fusion.py

PyTorch multi-modal neural network for late fusion of feature blocks.

Architecture:
  - Separate embedding tower per feature block (Linear -> BN -> ReLU -> Dropout)
  - Concat or attention-weighted merge of tower outputs
  - Shared classification head

Supports:
  - Learned attention weights across blocks
  - Auxiliary per-block classification losses (multi-task)
  - Training with phylogenetic eigenvectors as additional input
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── building blocks ───────────────────────────────────────────────────────────


class BlockTower(nn.Module):
    """
    Single-block embedding tower: projects a feature block into a shared
    embedding space.

    Architecture: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> ReLU
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BlockAttention(nn.Module):
    """
    Soft attention over block embeddings.
    Learns a scalar attention weight per block, then produces a
    weighted sum of block embedding vectors.
    """

    def __init__(self, n_blocks: int, embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(embedding_dim * n_blocks, n_blocks)

    def forward(self, block_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        block_embeddings : List of (batch, embedding_dim) tensors.

        Returns
        -------
        (batch, embedding_dim) weighted sum.
        """
        concat = torch.cat(block_embeddings, dim=1)  # (batch, n_blocks * emb_dim)
        weights = F.softmax(self.attention(concat), dim=1)  # (batch, n_blocks)
        stacked = torch.stack(block_embeddings, dim=1)  # (batch, n_blocks, emb_dim)
        attended = (stacked * weights.unsqueeze(2)).sum(dim=1)  # (batch, emb_dim)
        return attended


# ── main model ────────────────────────────────────────────────────────────────


class DeepFusionClassifier(nn.Module):
    """
    Multi-modal fusion classifier for fungal genome feature blocks.

    Parameters
    ----------
    block_dims      : Dict block_name -> input feature dimension.
    n_classes       : Number of output classes.
    hidden_dim      : Hidden dimension in each tower.
    embedding_dim   : Output dimension of each tower (shared embedding space).
    fusion          : 'concat' or 'attention'.
    dropout         : Dropout rate.
    aux_loss_weight : Weight for auxiliary per-block classification losses.
    """

    def __init__(
        self,
        block_dims: dict[str, int],
        n_classes: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        fusion: Literal["concat", "attention"] = "attention",
        dropout: float = 0.3,
        aux_loss_weight: float = 0.2,
    ):
        super().__init__()
        self.block_names = list(block_dims.keys())
        self.fusion = fusion
        self.aux_loss_weight = aux_loss_weight

        # Per-block towers
        self.towers = nn.ModuleDict(
            {
                name: BlockTower(dim, hidden_dim, embedding_dim, dropout)
                for name, dim in block_dims.items()
            }
        )

        # Auxiliary classifiers (one per block)
        self.aux_heads = nn.ModuleDict(
            {name: nn.Linear(embedding_dim, n_classes) for name in block_dims}
        )

        # Fusion layer
        n_blocks = len(block_dims)
        if fusion == "attention":
            self.attention = BlockAttention(n_blocks, embedding_dim)
            head_input_dim = embedding_dim
        else:  # concat
            head_input_dim = embedding_dim * n_blocks

        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(
        self,
        block_inputs: dict[str, torch.Tensor],
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        block_inputs      : Dict block_name -> (batch, feature_dim) tensor.
        return_embeddings : If True, also return block embeddings.

        Returns
        -------
        Dict with keys: 'logits', 'aux_logits', optionally 'embeddings'.
        """
        embeddings = {
            name: self.towers[name](block_inputs[name])
            for name in self.block_names
            if name in block_inputs
        }
        emb_list = [embeddings[name] for name in self.block_names if name in embeddings]

        # Auxiliary losses
        aux_logits = {name: self.aux_heads[name](emb) for name, emb in embeddings.items()}

        # Fusion
        if self.fusion == "attention":
            fused = self.attention(emb_list)
        else:
            fused = torch.cat(emb_list, dim=1)

        logits = self.classifier(fused)

        out = {"logits": logits, "aux_logits": aux_logits}
        if return_embeddings:
            out["embeddings"] = embeddings
        return out

    def compute_loss(
        self,
        outputs: dict,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute main + auxiliary losses."""
        main_loss = F.cross_entropy(outputs["logits"], y)
        aux_loss = torch.stack(
            [F.cross_entropy(logits, y) for logits in outputs["aux_logits"].values()]
        ).mean()
        return main_loss + self.aux_loss_weight * aux_loss


# ── training loop ─────────────────────────────────────────────────────────────


class DeepFusionTrainer:
    """
    Trainer for DeepFusionClassifier.

    Parameters
    ----------
    model       : Initialized DeepFusionClassifier.
    lr          : Learning rate.
    weight_decay: L2 regularization.
    n_epochs    : Training epochs.
    batch_size  : Mini-batch size.
    patience    : Early stopping patience (epochs).
    device      : 'cuda', 'cpu', or 'auto'.
    """

    def __init__(
        self,
        model: DeepFusionClassifier,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        device: str = "auto",
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

    def _blocks_to_tensors(
        self,
        block_data: dict[str, pd.DataFrame],
        genome_ids: list[str],
    ) -> dict[str, torch.Tensor]:
        """Convert feature DataFrames to tensors, aligned on genome_ids."""
        tensors = {}
        for name, df in block_data.items():
            aligned = df.loc[genome_ids].values.astype(np.float32)
            tensors[name] = torch.tensor(aligned, device=self.device)
        return tensors

    def fit(
        self,
        block_train: dict[str, pd.DataFrame],
        y_train: pd.Series,
        block_val: dict[str, pd.DataFrame] | None = None,
        y_val: pd.Series | None = None,
    ) -> list[dict]:
        """
        Train the model.

        Parameters
        ----------
        block_train : Dict block_name -> training feature DataFrame.
        y_train     : Training labels.
        block_val   : Optional validation feature DataFrames.
        y_val       : Optional validation labels.

        Returns
        -------
        List of epoch-level metrics dicts.
        """
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y_train.values)
        self.label_encoder_ = le

        genome_ids = y_train.index.tolist()
        X_tensors = self._blocks_to_tensors(block_train, genome_ids)
        y_tensor = torch.tensor(y_enc, dtype=torch.long, device=self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        history = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.model.train()

            # Mini-batch training
            indices = torch.randperm(len(genome_ids))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch_blocks = {name: t[batch_idx] for name, t in X_tensors.items()}
                batch_y = y_tensor[batch_idx]

                optimizer.zero_grad()
                outputs = self.model(batch_blocks)
                loss = self.model.compute_loss(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)
            metrics = {"epoch": epoch, "train_loss": avg_train_loss}

            # Validation
            if block_val is not None and y_val is not None:
                val_metrics = self._evaluate(block_val, y_val, le)
                metrics.update(val_metrics)

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0
                    self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            history.append(metrics)
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: {metrics}")

        # Restore best model
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        return history

    @torch.no_grad()
    def _evaluate(
        self,
        block_data: dict[str, pd.DataFrame],
        y: pd.Series,
        le,
    ) -> dict:
        self.model.eval()
        genome_ids = y.index.tolist()
        X_tensors = self._blocks_to_tensors(block_data, genome_ids)
        y_enc = torch.tensor(le.transform(y.values), dtype=torch.long, device=self.device)

        outputs = self.model(X_tensors)
        loss = self.model.compute_loss(outputs, y_enc)
        preds = outputs["logits"].argmax(dim=1)
        acc = (preds == y_enc).float().mean().item()

        return {"val_loss": loss.item(), "val_accuracy": acc}

    @torch.no_grad()
    def predict_proba(
        self,
        block_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Return softmax probability DataFrame for new genomes."""
        self.model.eval()
        genome_ids = list(next(iter(block_data.values())).index)
        X_tensors = self._blocks_to_tensors(block_data, genome_ids)
        outputs = self.model(X_tensors)
        proba = F.softmax(outputs["logits"], dim=1).cpu().numpy()
        return pd.DataFrame(
            proba,
            index=genome_ids,
            columns=self.label_encoder_.classes_,
        )

    def save(self, path: Path) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "label_encoder": self.label_encoder_,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.label_encoder_ = checkpoint["label_encoder"]
