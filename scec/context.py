"""Decoder-synchronous neural context prior for E8 spectral vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def laplace_magnitude_bits(magnitude: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Bits of the same nonnegative quantized Laplace family used by ANS."""
    scale = scale.clamp_min(0.02)
    p_zero = 1.0 - 0.5 * torch.exp(-0.5 / scale)
    p_positive = (
        0.5
        * torch.exp(-(magnitude - 0.5).clamp_min(0.5) / scale)
        * (-torch.expm1(-1.0 / scale))
    )
    probability = torch.where(magnitude < 0.5, p_zero, p_positive)
    return -torch.log2(probability.clamp_min(1e-9))


class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        correction = self.pointwise(F.gelu(self.depthwise(values)))
        return values + torch.sigmoid(self.scale) * correction


@dataclass
class E8ContextFields:
    support_probability: torch.Tensor
    magnitude_scale: torch.Tensor
    coset_probability: torch.Tensor
    sign_probability: torch.Tensor | None
    anchor_mask: torch.Tensor
    total_bits: torch.Tensor
    base_total_bits: torch.Tensor


class E8DecoderContextPrior(nn.Module):
    """Correct static E8 priors using only already decoded anchor vectors."""

    def __init__(
        self,
        support_probability: np.ndarray,
        magnitude_scale: np.ndarray,
        coset_probability: np.ndarray,
        steps: np.ndarray,
        block_size: int,
        anchor_period: int = 2,
        hidden: int = 48,
        sign_probability: np.ndarray | None = None,
        predict_sign: bool = False,
        resonance_centers_normalized: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        support = torch.from_numpy(np.asarray(support_probability, dtype=np.float32))
        scale = torch.from_numpy(np.asarray(magnitude_scale, dtype=np.float32))
        coset = torch.from_numpy(np.asarray(coset_probability, dtype=np.float32))
        self.register_buffer("base_support", support)
        self.register_buffer("base_scale", scale)
        self.register_buffer("base_coset", coset)
        self.register_buffer("steps", torch.from_numpy(np.asarray(steps, dtype=np.float32)))
        self.profiles, self.bands, coordinate_count = support.shape
        if coordinate_count != 8 or block_size % (self.bands * 8):
            raise ValueError("invalid E8 prior dimensions")
        self.block_size = int(block_size)
        self.groups = block_size // 8
        self.groups_per_band = self.groups // self.bands
        self.anchor_period = int(anchor_period)
        self.predict_sign = bool(predict_sign)
        if resonance_centers_normalized is None:
            # The five fixed frequency-feature centers of the 3 MHz
            # femtosecond-laser AE acquisition chain, normalized by Nyquist.
            resonance_centers_normalized = np.asarray(
                [199_200, 298_800, 497_300, 597_700, 697_300], dtype=np.float32
            ) / 1_500_000.0
        resonance_centers_normalized = np.asarray(
            resonance_centers_normalized, dtype=np.float32
        ).reshape(-1)
        if resonance_centers_normalized.size != 5:
            raise ValueError("exactly five normalized resonance centers are required")
        if np.any((resonance_centers_normalized < 0) | (resonance_centers_normalized > 1)):
            raise ValueError("normalized resonance centers must lie in [0, 1]")
        # Keep this as plain immutable metadata rather than a new state-dict
        # buffer so that all previously published checkpoints remain loadable.
        self.resonance_centers_normalized = tuple(
            float(value) for value in resonance_centers_normalized
        )
        if self.predict_sign:
            if sign_probability is None:
                sign_probability = np.full_like(support_probability, 0.5)
            self.register_buffer(
                "base_sign",
                torch.from_numpy(np.asarray(sign_probability, dtype=np.float32)),
            )
        anchor = (torch.arange(self.groups) % anchor_period == 0).view(1, self.groups, 1)
        self.register_buffer("fixed_anchor_mask", anchor)

        # 8 anchor values + 8 supports + coset + mask + 8 base logits +
        # 8 base log-scales + base coset logit + normalized frequency +
        # 5 fixed frequency-feature envelopes + normalized profile step.
        input_channels = 42
        self.input_projection = nn.Conv1d(input_channels, hidden, 1)
        self.blocks = nn.ModuleList(
            [DilatedContextBlock(hidden, dilation) for dilation in (1, 2, 4, 8)]
        )
        # The optional final eight channels form a decoder-synchronous sign
        # (spectral polarity/phase) head.  They are kept optional so published
        # 17-channel checkpoints remain exactly loadable.
        self.output_projection = nn.Conv1d(hidden, 25 if self.predict_sign else 17, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        frequency = (torch.arange(self.groups, dtype=torch.float32) * 8.0 + 3.5) / block_size
        self.register_buffer("normalized_frequency", frequency.view(1, self.groups, 1))
        centers = torch.tensor(self.resonance_centers_normalized, dtype=torch.float32)
        resonance = torch.exp(
            -0.5 * ((frequency[:, None] - centers[None, :]) / 0.018) ** 2
        )
        self.register_buffer("resonance_features", resonance.view(1, self.groups, 5))

    def base_fields(
        self, profile: torch.Tensor, groups: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand normalized-frequency priors to an arbitrary signal length."""
        groups = self.groups if groups is None else int(groups)
        if groups == self.groups:
            band_index = torch.arange(self.bands, device=profile.device).repeat_interleave(
                self.groups_per_band
            )
        else:
            band_index = torch.div(
                torch.arange(groups, device=profile.device) * self.bands,
                groups, rounding_mode="floor",
            ).clamp_max(self.bands - 1)
        support = self.base_support[profile][:, band_index, :]
        scale = self.base_scale[profile][:, band_index, :]
        coset = self.base_coset[profile][:, band_index]
        return support, scale, coset

    def base_sign_fields(self, profile: torch.Tensor, groups: int) -> torch.Tensor | None:
        if not self.predict_sign:
            return None
        band_index = torch.div(
            torch.arange(groups, device=profile.device) * self.bands,
            groups, rounding_mode="floor",
        ).clamp_max(self.bands - 1)
        return self.base_sign[profile][:, band_index, :]

    @staticmethod
    def _coordinate_bits(
        coordinates: torch.Tensor,
        probability: torch.Tensor,
        scale: torch.Tensor,
        sign_probability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonzero = coordinates != 0
        p = probability.clamp(1e-5, 1 - 1e-5)
        support_bits = torch.where(nonzero, -torch.log2(p), -torch.log2(1 - p))
        magnitude = (coordinates.abs().float() - 1.0).clamp_min(0.0)
        magnitude_bits = laplace_magnitude_bits(
            magnitude, scale
        )
        if sign_probability is None:
            sign_bits = torch.ones_like(magnitude_bits)
        else:
            p_sign = sign_probability.clamp(1e-5, 1 - 1e-5)
            sign_bits = torch.where(
                coordinates < 0, -torch.log2(p_sign), -torch.log2(1 - p_sign)
            )
        return support_bits + nonzero.float() * (sign_bits + magnitude_bits)

    @staticmethod
    def _coset_bits(coset: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        p = probability.clamp(1e-5, 1 - 1e-5)
        return torch.where(coset.bool(), -torch.log2(p), -torch.log2(1 - p))

    def forward(
        self,
        coordinates: torch.Tensor,
        coset: torch.Tensor,
        profile: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        compute_bits: bool = True,
    ) -> E8ContextFields:
        """Return entropy fields for B x groups x 8 integer coordinates."""
        batch, groups, _ = coordinates.shape
        base_support, base_scale, base_coset = self.base_fields(profile, groups)
        if context_mask is None:
            if groups == self.groups:
                anchor = self.fixed_anchor_mask.expand(batch, -1, -1)
            else:
                anchor = (
                    torch.arange(groups, device=coordinates.device)
                    .remainder(self.anchor_period).eq(0).view(1, groups, 1)
                    .expand(batch, -1, -1)
                )
        else:
            anchor = context_mask.to(device=coordinates.device, dtype=torch.bool)
            if anchor.ndim == 2:
                anchor = anchor.unsqueeze(-1)
            anchor = anchor.expand(batch, -1, -1)
            if anchor.shape != (batch, groups, 1):
                raise ValueError("context_mask must broadcast to B x groups x 1")
        anchor_float = anchor.to(coordinates.dtype)
        normalized = torch.sign(coordinates.float()) * torch.log1p(coordinates.abs().float()) / 4.0
        anchor_coordinates = normalized * anchor_float
        anchor_support = (coordinates != 0).float() * anchor_float
        anchor_coset = coset.float().unsqueeze(-1) * anchor_float
        step_feature = (
            torch.log(self.steps[profile]).sub(torch.log(self.steps).mean())
            / torch.log(self.steps).std(unbiased=False).clamp_min(1e-6)
        ).view(batch, 1, 1).expand(-1, groups, -1)
        if groups == self.groups:
            frequency = self.normalized_frequency
            resonance = self.resonance_features
        else:
            frequency = (
                (torch.arange(groups, dtype=torch.float32, device=coordinates.device) * 8.0 + 3.5)
                / (groups * 8.0)
            )
            centers = torch.tensor(
                self.resonance_centers_normalized,
                dtype=torch.float32, device=coordinates.device,
            )
            resonance = torch.exp(
                -0.5 * ((frequency[:, None] - centers[None, :]) / 0.018) ** 2
            ).view(1, groups, 5)
            frequency = frequency.view(1, groups, 1)
        features = torch.cat(
            [
                anchor_coordinates,
                anchor_support,
                anchor_coset,
                anchor_float,
                torch.logit(base_support.clamp(1e-4, 1 - 1e-4)),
                torch.log(base_scale.clamp_min(1e-4)),
                torch.logit(base_coset.clamp(1e-4, 1 - 1e-4)).unsqueeze(-1),
                frequency.expand(batch, -1, -1),
                resonance.expand(batch, -1, -1),
                step_feature,
            ],
            dim=-1,
        )
        hidden = F.gelu(self.input_projection(features.transpose(1, 2)))
        for block in self.blocks:
            hidden = block(hidden)
        corrections = self.output_projection(hidden).transpose(1, 2)
        support_delta = corrections[..., :8]
        scale_delta = corrections[..., 8:16]
        coset_delta = corrections[..., 16]
        base_sign = self.base_sign_fields(profile, groups)
        nonanchor = (~anchor).float()
        support = torch.sigmoid(
            torch.logit(base_support.clamp(1e-4, 1 - 1e-4))
            + nonanchor * 2.5 * torch.tanh(support_delta)
        ).clamp(1e-4, 1 - 1e-4)
        scale = base_scale * torch.exp(nonanchor * 2.2 * torch.tanh(scale_delta))
        base_coset_logit = torch.logit(base_coset.clamp(1e-4, 1 - 1e-4))
        coset_probability = torch.sigmoid(
            base_coset_logit + (~anchor[..., 0]).float() * 2.5 * torch.tanh(coset_delta)
        ).clamp(1e-4, 1 - 1e-4)
        sign_probability = None
        if self.predict_sign:
            sign_delta = corrections[..., 17:25]
            sign_probability = torch.sigmoid(
                torch.logit(base_sign.clamp(1e-4, 1 - 1e-4))
                + nonanchor * 2.5 * torch.tanh(sign_delta)
            ).clamp(1e-4, 1 - 1e-4)
        if compute_bits:
            total_bits = self._coordinate_bits(
                coordinates, support, scale, sign_probability
            ).sum(dim=(1, 2))
            total_bits += self._coset_bits(coset, coset_probability).sum(dim=1)
            base_bits = self._coordinate_bits(
                coordinates, base_support, base_scale, base_sign
            ).sum(dim=(1, 2))
            base_bits += self._coset_bits(coset, base_coset).sum(dim=1)
        else:
            # Deployment needs probability fields, not training diagnostics.
            total_bits = coordinates.new_zeros((batch,), dtype=torch.float32)
            base_bits = total_bits
        return E8ContextFields(
            support, scale.clamp_min(0.03), coset_probability, sign_probability,
            anchor, total_bits, base_bits,
        )
