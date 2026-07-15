"""Class-agnostic residual adapters for frozen vision-language features.

Unlike a linear probe, this module never owns a row per training class.  It
maps an image embedding back into the same embedding space, so the adapted
feature can still be compared with text embeddings for classes that were not
present while fitting the adapter.
"""

from __future__ import annotations

import math

import torch.nn.functional as functional
from torch import Tensor, nn


class ResidualFeatureAdapter(nn.Module):
    """A zero-initialised bottleneck residual map in CLIP feature space.

    Given a unit-normalised image feature ``x``, the adapter computes

    ``normalize(x + residual_scale * W_up(GELU(W_down(x))))``.

    ``W_up`` and its bias are initialised to zero.  The module is therefore an
    exact identity before training, including after the final normalisation.
    Its parameter count is independent of every dataset and class vocabulary.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        *,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.residual_scale = float(residual_scale)
        self.down = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden_dim, feature_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Deterministically resettable initialisation used by each fit trial."""

        self.down.reset_parameters()
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        """Restore the part of the adapter that guarantees identity output."""

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def residual(self, features: Tensor) -> Tensor:
        """Return the unnormalised residual for a batch of image features."""

        self._validate_features(features)
        return self.up(self.activation(self.down(features))) * self.residual_scale

    def forward(self, features: Tensor) -> Tensor:
        """Return unit-normalised adapted features in the original space."""

        self._validate_features(features)
        return functional.normalize(features + self.residual(features), dim=-1)

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [batch, {self.feature_dim}], "
                f"found {tuple(features.shape)}"
            )
        if not features.is_floating_point():
            raise TypeError("features must be floating point")


def similarity_logits(
    image_features: Tensor,
    text_prototypes: Tensor,
    *,
    logit_scale: float,
) -> Tensor:
    """Score adapted image features against an arbitrary text vocabulary."""

    if image_features.ndim != 2 or text_prototypes.ndim != 2:
        raise ValueError("image_features and text_prototypes must be matrices")
    if image_features.shape[1] != text_prototypes.shape[1]:
        raise ValueError("image and text feature dimensions must match")
    if not math.isfinite(logit_scale) or logit_scale <= 0:
        raise ValueError("logit_scale must be finite and positive")
    return float(logit_scale) * image_features @ text_prototypes.t()
