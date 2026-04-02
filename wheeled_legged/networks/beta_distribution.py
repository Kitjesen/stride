# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Beta distribution for bounded continuous action spaces.

Implements the Beta distribution for PPO as described in:
  Chou et al., "Improving Stochastic Policy Gradients in Continuous Control
  with Deep RL using the Beta Distribution", ICML 2017.

Used by the HLC navigation controller (arXiv:2405.01792) for bounded
velocity commands: vx∈[-1,2], vy∈[-0.75,0.75], ωz∈[-1.25,1.25].

The MLP outputs 2*output_dim values which are passed through Sigmoid
to produce (a1, a2) ∈ (0,1)², then:
    alpha = a1 * a2
    beta  = a2 * (1 - a1)
This ensures alpha, beta > 0 by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Beta

try:
    from rsl_rl.modules.distribution import Distribution
except ImportError:
    import os, sys
    # Search common locations for rsl_rl
    _candidates = [
        os.environ.get("RSL_RL_PATH", ""),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "thirdparty", "locomotion", "rsl_rl"),
        "D:/inovxio/thirdparty/locomotion/rsl_rl",
        "/root/autodl-tmp/thunder2/robot_lab/source/rsl_rl",
    ]
    for _p in _candidates:
        if _p and os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    from rsl_rl.modules.distribution import Distribution


# Numerical stability constants
_PARAM_EPS = 1e-4    # minimum alpha/beta to avoid degenerate distributions
_SAMPLE_EPS = 1e-6   # clamp samples away from 0/1 for log_prob stability


class BetaDistribution(Distribution):
    """Beta distribution for bounded action spaces.

    The MLP must output ``2 * output_dim`` values. These are reshaped to
    ``(batch, 2, output_dim)``, passed through Sigmoid, and converted to
    Beta distribution parameters via the Chou et al. parameterization.

    Actions are sampled in (0, 1) and must be rescaled externally to the
    desired physical range.
    """

    def __init__(self, output_dim: int, init_concentration: float = 2.0) -> None:
        """Initialize Beta distribution.

        Args:
            output_dim: Dimension of the action space.
            init_concentration: Initial bias for Sigmoid inputs so that
                alpha ≈ beta ≈ init_concentration / 2 (centered uniform-ish).
        """
        super().__init__(output_dim)
        self._init_concentration = init_concentration
        self._distribution: Beta | None = None
        self._alpha: torch.Tensor | None = None
        self._beta: torch.Tensor | None = None

        Beta.set_default_validate_args(False)

    # ── Interface required by rsl_rl ──────────────────────────────────────

    @property
    def input_dim(self) -> list[int]:
        """MLP must output (2, output_dim) — reshaped from flat 2*output_dim."""
        return [2, self.output_dim]

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update Beta params from MLP output of shape (..., 2, output_dim)."""
        a1 = torch.sigmoid(mlp_output[..., 0, :])  # (batch, output_dim)
        a2 = torch.sigmoid(mlp_output[..., 1, :])  # (batch, output_dim)
        self._alpha = (a1 * a2).clamp(min=_PARAM_EPS)
        self._beta = (a2 * (1.0 - a1)).clamp(min=_PARAM_EPS)
        self._distribution = Beta(self._alpha, self._beta)

    def sample(self) -> torch.Tensor:
        """Sample from Beta distribution, clamped for numerical safety."""
        raw = self._distribution.sample()
        return raw.clamp(_SAMPLE_EPS, 1.0 - _SAMPLE_EPS)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the mean of the Beta distribution: alpha / (alpha + beta)."""
        a1 = torch.sigmoid(mlp_output[..., 0, :])
        a2 = torch.sigmoid(mlp_output[..., 1, :])
        alpha = (a1 * a2).clamp(min=_PARAM_EPS)
        beta = (a2 * (1.0 - a1)).clamp(min=_PARAM_EPS)
        return alpha / (alpha + beta)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module for deterministic inference."""
        return _BetaMeanOutput()

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation of Beta: sqrt(alpha*beta / ((a+b)^2 * (a+b+1)))."""
        a, b = self._alpha, self._beta
        ab = a + b
        return torch.sqrt(a * b / (ab * ab * (ab + 1.0))).clamp(min=1e-8)

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._alpha, self._beta)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Log probability, with sample clamping for stability."""
        clamped = outputs.clamp(_SAMPLE_EPS, 1.0 - _SAMPLE_EPS)
        return self._distribution.log_prob(clamped).sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """KL(old || new) between two Beta distributions."""
        old_dist = Beta(old_params[0], old_params[1])
        new_dist = Beta(new_params[0], new_params[1])
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize last layer bias so initial distribution is roughly uniform."""
        # With sigmoid, bias=0 → a1=a2=0.5 → alpha=0.25, beta=0.25
        # This gives a U-shaped distribution. For a flatter start, use small positive bias.
        try:
            last_linear = mlp[-2]  # last Linear before activation
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, 0.5)
        except (IndexError, AttributeError):
            pass


class _BetaMeanOutput(nn.Module):
    """Export-friendly module: sigmoid → alpha/beta → mean."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        a1 = torch.sigmoid(mlp_output[..., 0, :])
        a2 = torch.sigmoid(mlp_output[..., 1, :])
        alpha = (a1 * a2).clamp(min=_PARAM_EPS)
        beta = (a2 * (1.0 - a1)).clamp(min=_PARAM_EPS)
        return alpha / (alpha + beta)


# ── Action rescaling utilities ────────────────────────────────────────────


def rescale_beta_actions(
    samples: torch.Tensor,
    low: torch.Tensor | list[float],
    high: torch.Tensor | list[float],
) -> torch.Tensor:
    """Rescale Beta samples from (0,1) to [low, high].

    For HLC (arXiv:2405.01792):
        low  = [-1.0, -0.75, -1.25]
        high = [ 2.0,  0.75,  1.25]
    """
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low, device=samples.device, dtype=samples.dtype)
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(high, device=samples.device, dtype=samples.dtype)
    return samples * (high - low) + low


def inverse_rescale_beta_actions(
    actions: torch.Tensor,
    low: torch.Tensor | list[float],
    high: torch.Tensor | list[float],
) -> torch.Tensor:
    """Inverse rescale physical actions back to (0,1) for log_prob computation."""
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low, device=actions.device, dtype=actions.dtype)
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(high, device=actions.device, dtype=actions.dtype)
    return ((actions - low) / (high - low)).clamp(_SAMPLE_EPS, 1.0 - _SAMPLE_EPS)
