# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the composable auxiliary losses (``rsl_rl/algorithms/losses.py``)."""

from __future__ import annotations

import pytest
import torch
from dataclasses import dataclass
from typing import Any

from rsl_rl.algorithms.losses import (
    AuxiliaryLoss,
    AuxLossContext,
    DecoderMatchingLoss,
    EncoderMatchingLoss,
    ImitationLoss,
    ReconstructionLoss,
    VAEKLLoss,
    build_scheduler,
    reject_legacy_loss_kwargs,
    resolve_aux_losses,
    resolve_loss_fn,
)


# ----------------------------------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------------------------------


class _StubActor:
    """Minimal duck-typed actor exposing only the hooks a given loss needs."""

    def __init__(
        self,
        has_decoder: bool = False,
        decoder_output: torch.Tensor | None = None,
        decoder_obs_groups: list[str] | None = None,
        encoder_state: torch.Tensor | None = None,
        vae_params: tuple[torch.Tensor, torch.Tensor] | None = None,
        decoder_inference: torch.Tensor | None = None,
    ) -> None:
        self.has_decoder = has_decoder
        self._decoder_output = decoder_output
        self.decoder_obs_groups = decoder_obs_groups or []
        self._encoder_state = encoder_state
        self._vae_params = vae_params
        self._decoder_inference = decoder_inference

    def get_decoder_output(self) -> torch.Tensor | None:
        return self._decoder_output

    def get_encoder_state(self) -> torch.Tensor | None:
        return self._encoder_state

    def get_vae_params(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._vae_params

    def get_decoder_inference(self, latent: torch.Tensor) -> torch.Tensor:
        # A simple deterministic "decoder": scale the latent so the test can predict the output.
        return 2.0 * latent


@dataclass
class _StubBatch:
    """Duck-typed mini-batch carrying only the fields losses read."""

    observations: Any = None
    privileged_actions: torch.Tensor | None = None
    encoder_state: torch.Tensor | None = None
    privileged_encoder_state: torch.Tensor | None = None


def _ctx(actor: _StubActor, batch: _StubBatch, teacher: _StubActor | None = None, dist_mean: torch.Tensor | None = None) -> AuxLossContext:
    """Build an AuxLossContext from stubs."""
    return AuxLossContext(
        actor=actor,
        critic=None,
        teacher=teacher,
        batch=batch,
        distribution_params=(dist_mean,) if dist_mean is not None else (),
        original_batch_size=0 if dist_mean is None else dist_mean.shape[0],
        device="cpu",
    )


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


class TestResolveLossFn:
    """Tests for ``resolve_loss_fn``."""

    def test_known_types(self) -> None:
        assert resolve_loss_fn("mse") is torch.nn.functional.mse_loss
        assert resolve_loss_fn("huber") is torch.nn.functional.huber_loss

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss_type"):
            resolve_loss_fn("l1")


class TestBuildScheduler:
    """Tests for ``build_scheduler``."""

    def test_none_returns_none(self) -> None:
        assert build_scheduler(1.0, None) is None

    def test_constant(self) -> None:
        sched = build_scheduler(0.7, {"mode": "constant"})
        assert sched(0) == 0.7
        assert sched(10_000) == 0.7

    def test_step(self) -> None:
        sched = build_scheduler(1.0, {"mode": "step", "final_step": 100, "final_value": 0.1})
        assert sched(99) == 1.0
        assert sched(100) == 0.1

    def test_linear_interpolates_and_clamps(self) -> None:
        sched = build_scheduler(
            1.0, {"mode": "linear", "initial_step": 0, "final_step": 100, "final_value": 0.0}
        )
        assert sched(0) == pytest.approx(1.0)
        assert sched(50) == pytest.approx(0.5)
        assert sched(100) == pytest.approx(0.0)
        assert sched(200) == pytest.approx(0.0)  # clamped after final_step

    def test_linear_initial_value_override(self) -> None:
        # The PPO-weight curriculum starts below the default coefficient.
        sched = build_scheduler(
            1.0,
            {"mode": "linear", "initial_value": 0.0, "initial_step": 0, "final_step": 100, "final_value": 0.9},
        )
        assert sched(0) == pytest.approx(0.0)
        assert sched(100) == pytest.approx(0.9)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown weight_schedule mode"):
            build_scheduler(1.0, {"mode": "cosine"})


class TestRejectLegacyLossKwargs:
    """Tests for ``reject_legacy_loss_kwargs``."""

    def test_clean_config_passes(self) -> None:
        reject_legacy_loss_kwargs({"clip_param": 0.2, "aux_losses": []})  # no raise

    @pytest.mark.parametrize(
        "key", ["decoder_loss_coef", "kl_loss_coef", "imitation_loss_coef", "loss_type", "loss_schedule"]
    )
    def test_legacy_key_raises(self, key: str) -> None:
        with pytest.raises(ValueError, match="removed loss kwargs"):
            reject_legacy_loss_kwargs({key: 1.0})


class TestResolveAuxLosses:
    """Tests for ``resolve_aux_losses``."""

    def test_empty(self) -> None:
        cfg: dict = {}
        assert resolve_aux_losses(cfg) == []

    def test_builds_instances_and_pops_key(self) -> None:
        cfg = {
            "aux_losses": [
                {"class_name": "rsl_rl.algorithms.losses.ReconstructionLoss", "coef": 0.5, "loss_type": "huber"},
                {"class_name": "rsl_rl.algorithms.losses.VAEKLLoss", "coef": 0.01, "kl_clip": 0.2},
            ]
        }
        losses = resolve_aux_losses(cfg)
        assert "aux_losses" not in cfg  # key consumed
        assert isinstance(losses[0], ReconstructionLoss)
        assert losses[0].coef == 0.5
        assert losses[0].loss_fn is torch.nn.functional.huber_loss
        assert isinstance(losses[1], VAEKLLoss)
        assert losses[1].kl_clip == 0.2

    def test_does_not_mutate_caller_specs(self) -> None:
        spec = {"class_name": "rsl_rl.algorithms.losses.ImitationLoss", "coef": 1.0}
        cfg = {"aux_losses": [spec]}
        resolve_aux_losses(cfg)
        assert spec == {"class_name": "rsl_rl.algorithms.losses.ImitationLoss", "coef": 1.0}


# ----------------------------------------------------------------------------------------------
# Coefficient / scheduling on the base class
# ----------------------------------------------------------------------------------------------


class TestCoefficient:
    """Tests for ``AuxiliaryLoss.coefficient``."""

    def test_constant_when_no_schedule(self) -> None:
        loss = ImitationLoss(coef=0.3)
        assert loss.coefficient(0) == 0.3
        assert loss.coefficient(1000) == 0.3

    def test_scheduled(self) -> None:
        loss = ImitationLoss(
            coef=1.0,
            weight_schedule={"mode": "linear", "initial_step": 0, "final_step": 100, "final_value": 0.1},
        )
        assert loss.coefficient(0) == pytest.approx(1.0)
        assert loss.coefficient(100) == pytest.approx(0.1)


# ----------------------------------------------------------------------------------------------
# Per-loss applicability + math
# ----------------------------------------------------------------------------------------------


class TestReconstructionLoss:
    """Tests for ``ReconstructionLoss``."""

    def test_not_applicable_without_decoder(self) -> None:
        loss = ReconstructionLoss()
        ctx = _ctx(_StubActor(has_decoder=False), _StubBatch())
        assert not loss.is_applicable(ctx)

    def test_not_applicable_when_output_none(self) -> None:
        loss = ReconstructionLoss()
        ctx = _ctx(_StubActor(has_decoder=True, decoder_output=None), _StubBatch())
        assert not loss.is_applicable(ctx)

    def test_compute_matches_loss_fn(self) -> None:
        recon = torch.zeros(3, 2)
        target = torch.ones(3, 2)
        actor = _StubActor(has_decoder=True, decoder_output=recon, decoder_obs_groups=["a", "b"])
        batch = _StubBatch(observations={"a": torch.ones(3, 1), "b": torch.ones(3, 1)})
        loss = ReconstructionLoss(loss_type="mse")
        ctx = _ctx(actor, batch)
        assert loss.is_applicable(ctx)
        # target = cat([ones(3,1), ones(3,1)]) = ones(3,2); mse(zeros, ones) = 1.0
        assert loss.compute(ctx).item() == pytest.approx(1.0)


class TestVAEKLLoss:
    """Tests for ``VAEKLLoss``."""

    def test_not_applicable_without_vae_params(self) -> None:
        loss = VAEKLLoss()
        # get_vae_params() returns None before the first forward pass (non-VAE backend).
        ctx = _ctx(_StubActor(vae_params=None), _StubBatch())
        assert not loss.is_applicable(ctx)

    def test_kl_zero_at_standard_normal(self) -> None:
        # mu = 0, log_var = 0 (sigma^2 = 1) => KL per dim = 0.
        mu = torch.zeros(4, 3)
        log_var = torch.zeros(4, 3)
        loss = VAEKLLoss(kl_clip=0.0)
        ctx = _ctx(_StubActor(vae_params=(mu, log_var)), _StubBatch())
        assert loss.is_applicable(ctx)
        assert loss.compute(ctx).item() == pytest.approx(0.0, abs=1e-6)

    def test_kl_free_nats_clip(self) -> None:
        # With mu=0, log_var=0 the per-dim KL is 0; clipping at 0.5 floors every dim to 0.5.
        mu = torch.zeros(4, 3)
        log_var = torch.zeros(4, 3)
        loss = VAEKLLoss(kl_clip=0.5)
        ctx = _ctx(_StubActor(vae_params=(mu, log_var)), _StubBatch())
        assert loss.compute(ctx).item() == pytest.approx(0.5)


class TestImitationLoss:
    """Tests for ``ImitationLoss``."""

    def test_not_applicable_without_teacher(self) -> None:
        loss = ImitationLoss()
        ctx = _ctx(_StubActor(), _StubBatch(privileged_actions=torch.ones(3, 2)), teacher=None)
        assert not loss.is_applicable(ctx)

    def test_compute(self) -> None:
        mean = torch.zeros(3, 2)
        priv = torch.ones(3, 2)
        loss = ImitationLoss(loss_type="mse")
        ctx = _ctx(_StubActor(), _StubBatch(privileged_actions=priv), teacher=_StubActor(), dist_mean=mean)
        assert loss.is_applicable(ctx)
        assert loss.compute(ctx).item() == pytest.approx(1.0)


class TestEncoderMatchingLoss:
    """Tests for ``EncoderMatchingLoss``."""

    def test_not_applicable_without_encoder_states(self) -> None:
        loss = EncoderMatchingLoss()
        ctx = _ctx(_StubActor(), _StubBatch(encoder_state=torch.ones(3, 2), privileged_encoder_state=None))
        assert not loss.is_applicable(ctx)

    def test_compute(self) -> None:
        student_enc = torch.zeros(3, 2)
        priv_enc = torch.ones(3, 2)
        actor = _StubActor(encoder_state=student_enc)
        batch = _StubBatch(encoder_state=student_enc, privileged_encoder_state=priv_enc)
        loss = EncoderMatchingLoss(loss_type="mse")
        ctx = _ctx(actor, batch, teacher=_StubActor())
        assert loss.is_applicable(ctx)
        assert loss.compute(ctx).item() == pytest.approx(1.0)


class TestDecoderMatchingLoss:
    """Tests for ``DecoderMatchingLoss``."""

    def test_compute_uses_teacher_decoder(self) -> None:
        student_enc = torch.zeros(3, 2)
        priv_enc = torch.ones(3, 2)
        actor = _StubActor(encoder_state=student_enc)
        teacher = _StubActor()  # get_decoder_inference(x) = 2x
        batch = _StubBatch(encoder_state=student_enc, privileged_encoder_state=priv_enc)
        loss = DecoderMatchingLoss(loss_type="mse")
        ctx = _ctx(actor, batch, teacher=teacher)
        assert loss.is_applicable(ctx)
        # student_dec = 2*0 = 0; teacher_dec = 2*1 = 2; mse(0, 2) = 4.
        assert loss.compute(ctx).item() == pytest.approx(4.0)
