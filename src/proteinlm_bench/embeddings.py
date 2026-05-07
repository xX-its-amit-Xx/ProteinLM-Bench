"""Protein language model embeddings.

Two backends are exposed:

* ``MockEmbedder``  - lightweight deterministic hash-based embeddings used for
  CI, smoke tests, and the ``--mock-embeddings`` CLI flag. No model weights are
  downloaded; CPU-only and ~instantaneous.
* ``HuggingFaceEmbedder`` - wraps any HuggingFace ``AutoModel`` (tested with
  ``facebook/esm2_t6_8M_UR50D`` and ``Rostlab/prot_bert``) and pools per-residue
  hidden states into a fixed-size sequence embedding.

Both implement the same ``embed_sequences`` interface, so the rest of the
pipeline does not need to know which backend produced the vectors.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .utils import PathLike, ensure_dir, get_logger

logger = get_logger(__name__)


@dataclass
class EmbedderConfig:
    """Backend-agnostic embedder settings."""

    backend: str = "mock"
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    pooling: str = "mean"           # mean | cls | mut_site
    device: str = "cpu"
    batch_size: int = 4
    max_length: int = 1024
    cache_dir: Optional[str] = None
    embedding_dim: int = 128         # only consulted for the mock backend


class BaseEmbedder:
    """Abstract embedder interface."""

    embedding_dim: int

    def embed_sequences(
        self,
        sequences: Sequence[str],
        *,
        mutation_positions: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> np.ndarray:
        """Embed a batch of sequences and return an ``(n, d)`` array.

        Parameters
        ----------
        sequences:
            Iterable of amino-acid strings.
        mutation_positions:
            Optional per-sequence list of mutated positions (1-indexed). Only
            used when ``pooling="mut_site"``.
        """
        raise NotImplementedError


class MockEmbedder(BaseEmbedder):
    """Deterministic hash-derived embeddings.

    The vector for each sequence is built from a SHA-256 digest of the
    sequence string. Identical sequences map to identical vectors, and small
    edits produce structured perturbations — which is enough to give the
    downstream regressors a non-trivial signal during smoke tests.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = int(embedding_dim)

    def _sequence_vector(self, seq: str) -> np.ndarray:
        seed_bytes = hashlib.sha256(seq.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(self.embedding_dim).astype(np.float32)

        # Add a few interpretable features so single-residue edits produce
        # smooth changes (rather than fully random ones, which would make the
        # downstream regressors hopeless).
        aa_counts = np.zeros(26, dtype=np.float32)
        for ch in seq:
            i = ord(ch.upper()) - ord("A")
            if 0 <= i < 26:
                aa_counts[i] += 1.0
        if len(seq) > 0:
            aa_counts /= len(seq)

        d = self.embedding_dim
        if d >= 26:
            base[:26] = 0.5 * base[:26] + aa_counts
        else:
            base = 0.5 * base + aa_counts[:d]
        return base.astype(np.float32)

    def embed_sequences(
        self,
        sequences: Sequence[str],
        *,
        mutation_positions: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> np.ndarray:
        return np.stack([self._sequence_vector(s) for s in sequences], axis=0)


class HuggingFaceEmbedder(BaseEmbedder):
    """Wraps a HuggingFace transformer for protein sequence embedding.

    Tested model identifiers:
      * ``facebook/esm2_t6_8M_UR50D`` (small ESM2; CPU-friendly)
      * ``Rostlab/prot_bert`` (whitespace-tokenised)

    Embeddings are mean-pooled over the residue axis by default; ``"cls"``
    pooling returns the first token's hidden state and ``"mut_site"`` averages
    the hidden states at the mutated positions only (1-indexed).
    """

    def __init__(self, config: EmbedderConfig) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise ImportError(
                "The HuggingFace backend requires `transformers` and `torch`. "
                "Install with `pip install transformers torch`, or use "
                "`backend: mock` / `--mock-embeddings`."
            ) from exc

        self._torch = torch
        self.config = config
        cache_dir = config.cache_dir
        if cache_dir is not None:
            ensure_dir(cache_dir)

        logger.info("Loading HuggingFace tokenizer/model: %s", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, cache_dir=cache_dir, do_lower_case=False
        )
        self.model = AutoModel.from_pretrained(config.model_name, cache_dir=cache_dir)
        self.model.eval()
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.embedding_dim = int(self.model.config.hidden_size)
        self._is_prot_bert = "prot_bert" in config.model_name.lower()

    def _format_sequence(self, seq: str) -> str:
        if self._is_prot_bert:
            # ProtBert expects whitespace-separated residues and uses ambiguous
            # AA mapping for U/Z/O/B → X.
            cleaned = seq.upper()
            for ch in "UZOB":
                cleaned = cleaned.replace(ch, "X")
            return " ".join(cleaned)
        return seq

    def _pool(
        self,
        hidden: "object",                # torch.Tensor (n, L, d)
        attention_mask: "object",         # torch.Tensor (n, L)
        mutation_positions: Optional[Sequence[Optional[Sequence[int]]]],
        offset: int,
    ) -> "object":
        torch = self._torch
        pooling = self.config.pooling
        if pooling == "cls":
            return hidden[:, 0, :]
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom
        if pooling == "mut_site":
            if mutation_positions is None:
                raise ValueError("pooling='mut_site' requires mutation_positions")
            pooled = []
            for i, positions in enumerate(mutation_positions):
                if not positions:
                    mask = attention_mask[i].unsqueeze(-1).float()
                    summed = (hidden[i] * mask).sum(dim=0)
                    denom = mask.sum().clamp(min=1.0)
                    pooled.append(summed / denom)
                    continue
                idx = [min(p - 1 + offset, hidden.shape[1] - 1) for p in positions]
                pooled.append(hidden[i, idx, :].mean(dim=0))
            return torch.stack(pooled, dim=0)
        raise ValueError(f"Unknown pooling: {pooling!r}")

    def embed_sequences(
        self,
        sequences: Sequence[str],
        *,
        mutation_positions: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> np.ndarray:
        torch = self._torch
        all_embeds: List[np.ndarray] = []
        bs = max(1, int(self.config.batch_size))
        # Special-token offset: most BERT-family tokenizers prepend [CLS], so
        # residue at protein position p sits at token index p (0-indexed in
        # hidden_states) — i.e. an offset of +1 from 1-indexed coordinates.
        offset = 1 if self._is_prot_bert or True else 0

        with torch.no_grad():
            for start in range(0, len(sequences), bs):
                batch_seqs = list(sequences[start : start + bs])
                batch_positions = (
                    list(mutation_positions[start : start + bs])
                    if mutation_positions is not None
                    else None
                )
                formatted = [self._format_sequence(s) for s in batch_seqs]
                tokens = self.tokenizer(
                    formatted,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.model(**tokens)
                hidden = outputs.last_hidden_state
                pooled = self._pool(
                    hidden,
                    tokens["attention_mask"],
                    batch_positions,
                    offset=offset,
                )
                all_embeds.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeds, axis=0)


def build_embedder(config: EmbedderConfig) -> BaseEmbedder:
    """Factory returning the appropriate embedder for a given config."""
    backend = config.backend.lower()
    if backend == "mock":
        return MockEmbedder(embedding_dim=config.embedding_dim)
    if backend in {"huggingface", "hf"}:
        return HuggingFaceEmbedder(config)
    raise ValueError(
        f"Unknown embeddings backend: {config.backend!r}. "
        "Expected one of: 'mock', 'huggingface'."
    )


def embedder_from_dict(cfg: dict, *, mock_override: bool = False) -> BaseEmbedder:
    """Build an embedder directly from the YAML ``embeddings`` section."""
    backend = "mock" if mock_override else cfg.get("backend", "mock")
    embedder_cfg = EmbedderConfig(
        backend=backend,
        model_name=cfg.get("model_name", "facebook/esm2_t6_8M_UR50D"),
        pooling=cfg.get("pooling", "mean"),
        device=cfg.get("device", "cpu"),
        batch_size=int(cfg.get("batch_size", 4)),
        max_length=int(cfg.get("max_length", 1024)),
        cache_dir=cfg.get("cache_dir"),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
    )
    return build_embedder(embedder_cfg)


__all__ = [
    "BaseEmbedder",
    "MockEmbedder",
    "HuggingFaceEmbedder",
    "EmbedderConfig",
    "build_embedder",
    "embedder_from_dict",
]
