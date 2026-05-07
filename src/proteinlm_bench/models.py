"""Lightweight downstream regressors used on top of frozen PLM embeddings.

The benchmark deliberately keeps these heads small — the goal is to compare
embedding quality, not to engineer the best possible regression head. All
models share a sklearn-style ``fit`` / ``predict`` interface so the rest of
the pipeline can iterate over them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .utils import get_logger

logger = get_logger(__name__)


class BaseRegressor:
    """Common interface for downstream regressors."""

    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRegressor":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RidgeRegressor(BaseRegressor):
    """Wrapper around scikit-learn ``Ridge`` with sensible defaults."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        self.model = Ridge(alpha=alpha, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.float64)


class RandomForestRegressorWrapper(BaseRegressor):
    """Random forest regressor; supports per-tree variance for uncertainty."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 0,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorWrapper":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.float64)

    def predict_with_variance(self, X: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """Per-tree mean and variance, used as a cheap uncertainty estimate."""
        per_tree = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=0)
        return per_tree.mean(axis=0), per_tree.var(axis=0)


class _MLP:
    """Internal: small PyTorch MLP. Constructed lazily inside MLPRegressor."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float):
        import torch.nn as nn

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        return self.net(x).squeeze(-1)


class MLPRegressor(BaseRegressor):
    """Shallow MLP regressor implemented in PyTorch.

    Standardises features internally (the embeddings can have wildly different
    scales between PLMs, and the small example dataset is sensitive to this).
    """

    name = "mlp"

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device_name = device
        self.seed = int(seed)
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._mlp: Optional[_MLP] = None

    def _standardize(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        if fit:
            self._mu = X.mean(axis=0, keepdims=True)
            self._sigma = X.std(axis=0, keepdims=True)
            self._sigma = np.where(self._sigma < 1e-8, 1.0, self._sigma)
        assert self._mu is not None and self._sigma is not None
        return (X - self._mu) / self._sigma

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        import torch

        torch.manual_seed(self.seed)
        device = torch.device(self.device_name)
        Xs = self._standardize(X.astype(np.float32), fit=True)
        ys = y.astype(np.float32).reshape(-1)

        self._mlp = _MLP(Xs.shape[1], self.hidden_dims, self.dropout)
        self._mlp.net.to(device)
        optim = torch.optim.Adam(
            self._mlp.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()

        X_t = torch.from_numpy(Xs).to(device)
        y_t = torch.from_numpy(ys).to(device)
        n = X_t.shape[0]
        idx = torch.arange(n)

        self._mlp.net.train()
        for epoch in range(self.epochs):
            perm = idx[torch.randperm(n)]
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = perm[start : start + self.batch_size]
                optim.zero_grad()
                pred = self._mlp(X_t[batch])
                loss = loss_fn(pred, y_t[batch])
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().cpu()) * batch.numel()
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.debug(
                    "MLP epoch %d/%d  loss=%.4f", epoch + 1, self.epochs, epoch_loss / max(1, n)
                )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._mlp is None:
            raise RuntimeError("MLPRegressor.predict called before fit().")
        Xs = self._standardize(X.astype(np.float32), fit=False)
        device = next(self._mlp.net.parameters()).device
        self._mlp.net.eval()
        with torch.no_grad():
            preds = self._mlp(torch.from_numpy(Xs).to(device)).cpu().numpy()
        return preds.astype(np.float64)


@dataclass
class EnsembleResult:
    """Mean prediction and per-member predictions from an ensemble fit."""

    mean: np.ndarray
    member_predictions: np.ndarray  # shape (n_members, n_samples)

    @property
    def variance(self) -> np.ndarray:
        return self.member_predictions.var(axis=0)


def fit_ensemble(
    factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_members: int = 5,
    base_seed: int = 0,
) -> EnsembleResult:
    """Fit an ensemble of regressors with bootstrap-resampled training data.

    ``factory(seed)`` should return a freshly constructed regressor for the
    given seed. Predictions are returned for ``X_test`` only — that is the
    common case for the benchmark and avoids storing test embeddings twice.
    """
    rng = np.random.default_rng(base_seed)
    n = len(X_train)
    members = []
    for k in range(n_members):
        idx = rng.integers(0, n, size=n)
        model = factory(int(rng.integers(0, 2**31 - 1)))
        model.fit(X_train[idx], y_train[idx])
        members.append(model.predict(X_test))
    arr = np.stack(members, axis=0)
    return EnsembleResult(mean=arr.mean(axis=0), member_predictions=arr)


def build_models_from_config(cfg: dict, *, seed: int = 0) -> Dict[str, BaseRegressor]:
    """Instantiate the set of models enabled in the YAML ``models`` section."""
    models: Dict[str, BaseRegressor] = {}
    ridge_cfg = cfg.get("ridge", {})
    if ridge_cfg.get("enabled", True):
        models["ridge"] = RidgeRegressor(alpha=float(ridge_cfg.get("alpha", 1.0)))
    rf_cfg = cfg.get("random_forest", {})
    if rf_cfg.get("enabled", True):
        models["random_forest"] = RandomForestRegressorWrapper(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth"),
            random_state=seed,
        )
    mlp_cfg = cfg.get("mlp", {})
    if mlp_cfg.get("enabled", True):
        models["mlp"] = MLPRegressor(
            hidden_dims=mlp_cfg.get("hidden_dims", [128, 64]),
            dropout=float(mlp_cfg.get("dropout", 0.1)),
            lr=float(mlp_cfg.get("lr", 1e-3)),
            weight_decay=float(mlp_cfg.get("weight_decay", 1e-4)),
            epochs=int(mlp_cfg.get("epochs", 50)),
            batch_size=int(mlp_cfg.get("batch_size", 32)),
            seed=seed,
        )
    return models


__all__ = [
    "BaseRegressor",
    "RidgeRegressor",
    "RandomForestRegressorWrapper",
    "MLPRegressor",
    "EnsembleResult",
    "fit_ensemble",
    "build_models_from_config",
]
