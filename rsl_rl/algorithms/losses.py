# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composable auxiliary losses for the PPO algorithm family.

Auxiliary losses are optional, self-describing training-loss terms that are added to the PPO
objective during ``update()``. Each loss:

- has a :attr:`~AuxiliaryLoss.name` used as its key in the returned loss dict (for logging),
- declares :meth:`~AuxiliaryLoss.is_applicable` (a *self-guard* — it skips silently when the
  model backend lacks the hook it needs, e.g. a reconstruction loss on a plain-MLP actor),
- computes a *raw, unweighted* scalar via :meth:`~AuxiliaryLoss.compute`,
- exposes a :meth:`~AuxiliaryLoss.coefficient`, optionally driven by a weight schedule.

Losses are selected explicitly by config (a list of specs resolved by :func:`resolve_aux_losses`)
*and* self-guard, so the same list can be reused across MLP / AutoEncoder / VAE / RNN / TCN
backends — inapplicable terms become no-ops rather than errors.

See ``rsl_rl/algorithms/aux_loss_design.md`` for the full design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable

from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


def resolve_loss_fn(loss_type: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Resolve a regression loss function by name.

    Centralizes the ``mse``/``huber`` choice that was previously re-declared in every algorithm
    class.

    Args:
        loss_type: One of ``"mse"`` or ``"huber"``.

    Returns:
        The corresponding ``torch.nn.functional`` loss callable.
    """
    table = {
        "mse": nn.functional.mse_loss,
        "huber": nn.functional.huber_loss,
    }
    if loss_type not in table:
        raise ValueError(f"Unknown loss_type {loss_type!r}. Supported types are: {list(table)}")
    return table[loss_type]


def build_scheduler(initial_value: float, schedule: dict | None) -> Callable[[int], float] | None:
    """Build a coefficient schedule callable ``step -> value`` (or ``None`` for a constant coef).

    Mirrors the constant/step/linear schedules used by RND (:mod:`rsl_rl.extensions.rnd`).

    Args:
        initial_value: Value before / at the start of the schedule (the loss coefficient).
        schedule: Schedule config dict with a ``"mode"`` key, or ``None`` for a constant value.
            Supported modes:

            - ``"constant"``: always ``initial_value``.
            - ``"step"``: ``initial_value`` until ``final_step``, then ``final_value``.
            - ``"linear"``: linearly interpolate from ``initial_value`` (at ``initial_step``) to
              ``final_value`` (at ``final_step``).

    Returns:
        A callable mapping iteration -> coefficient, or ``None`` if ``schedule`` is ``None``.
    """
    if schedule is None:
        return None

    mode = schedule["mode"]

    if mode == "constant":
        return lambda step: initial_value

    if mode == "step":
        final_step = schedule["final_step"]
        final_value = schedule["final_value"]
        return lambda step: initial_value if step < final_step else final_value

    if mode == "linear":
        initial_step = schedule["initial_step"]
        final_step = schedule["final_step"]
        final_value = schedule["final_value"]

        def _linear(step: int) -> float:
            if step < initial_step:
                return initial_value
            if step > final_step:
                return final_value
            frac = (step - initial_step) / (final_step - initial_step)
            return initial_value + (final_value - initial_value) * frac

        return _linear

    raise ValueError(f"Unknown weight_schedule mode {mode!r}. Supported: 'constant', 'step', 'linear'.")


# ----------------------------------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------------------------------


@dataclass
class AuxLossContext:
    """Everything an auxiliary loss may need during a single ``update()`` mini-batch.

    Bundled into one object so loss signatures stay stable as new losses are added.
    """

    actor: MLPModel
    """The student / actor model (re-run on the current batch before losses are computed)."""

    critic: MLPModel
    """The critic model."""

    teacher: MLPModel | None
    """The frozen teacher model, or ``None`` for non-distillation algorithms."""

    batch: Any
    """The current mini-batch (a :class:`~rsl_rl.storage.RolloutStorage` batch)."""

    distribution_params: tuple[torch.Tensor, ...]
    """Student output-distribution parameters, sliced to ``original_batch_size`` (mean is ``[0]``)."""

    original_batch_size: int
    """Mini-batch size before symmetry augmentation."""

    device: str
    """Torch device string."""


# ----------------------------------------------------------------------------------------------
# Base classes
# ----------------------------------------------------------------------------------------------


class AuxiliaryLoss:
    """Base class for an auxiliary training-loss term."""

    name: str = "aux"
    """Key under which this loss is reported in the algorithm's loss dict."""

    def __init__(self, coef: float = 1.0, weight_schedule: dict | None = None) -> None:
        """Initialize the loss.

        Args:
            coef: Scalar weight applied to the raw loss when added to the total objective.
            weight_schedule: Optional schedule dict (see :func:`build_scheduler`) that overrides
                ``coef`` as a function of the training iteration.
        """
        self.coef = coef
        self._scheduler = build_scheduler(coef, weight_schedule)

    def is_applicable(self, ctx: AuxLossContext) -> bool:
        """Return whether this loss can be computed for the given context (self-guard)."""
        return True

    def compute(self, ctx: AuxLossContext) -> torch.Tensor:
        """Compute the raw (unweighted) loss scalar."""
        raise NotImplementedError

    def coefficient(self, iteration: int) -> float:
        """Return the (possibly scheduled) coefficient for the given training iteration."""
        return self._scheduler(iteration) if self._scheduler is not None else self.coef


class _RegressionAuxiliaryLoss(AuxiliaryLoss):
    """Base for auxiliary losses that regress one tensor onto another (``mse``/``huber``)."""

    def __init__(self, coef: float = 1.0, loss_type: str = "mse", weight_schedule: dict | None = None) -> None:
        super().__init__(coef=coef, weight_schedule=weight_schedule)
        self.loss_fn = resolve_loss_fn(loss_type)


# ----------------------------------------------------------------------------------------------
# Concrete losses
# ----------------------------------------------------------------------------------------------


class ReconstructionLoss(_RegressionAuxiliaryLoss):
    """Auto-encoder reconstruction loss.

    Trains the actor's decoder to reconstruct its encoder observations from the encoder latent.
    Applicable to actors with a decoder (``has_decoder is True``, e.g. ``MLPAutoEncoderModel`` /
    ``MLPVAEModel``). Extracted from ``PPOAE.update``.
    """

    name = "decoder"

    def is_applicable(self, ctx: AuxLossContext) -> bool:
        return getattr(ctx.actor, "has_decoder", False) and ctx.actor.get_decoder_output() is not None

    def compute(self, ctx: AuxLossContext) -> torch.Tensor:
        decoder_output = ctx.actor.get_decoder_output()
        # Reconstruct the (normalized) encoder observations used as the decoder target.
        with torch.no_grad():
            decoder_target = torch.cat(
                [ctx.batch.observations[g] for g in ctx.actor.decoder_obs_groups],
                dim=-1,
            )
        return self.loss_fn(decoder_output, decoder_target)


class VAEKLLoss(AuxiliaryLoss):
    r"""beta-VAE latent KL loss: :math:`\mathrm{KL}(q(z\mid x)\,\|\,\mathcal{N}(0, I))`.

    Applicable to actors exposing ``get_vae_params() -> (mu, log_var)`` (e.g. ``MLPVAEModel``).
    Extracted from ``PPOVAE.update``. Note this loss has no ``loss_type`` — the KL form is fixed.
    """

    name = "kl"

    def __init__(self, coef: float = 1.0, kl_clip: float = 0.0, weight_schedule: dict | None = None) -> None:
        """Initialize the VAE KL loss.

        Args:
            coef: Weight of the KL loss.
            kl_clip: Free-nats tolerance. Only per-dimension KL above this threshold is penalized
                (prevents posterior collapse). ``0.0`` disables clipping (standard VAE).
            weight_schedule: Optional coefficient schedule.
        """
        super().__init__(coef=coef, weight_schedule=weight_schedule)
        self.kl_clip = kl_clip

    def is_applicable(self, ctx: AuxLossContext) -> bool:
        get_vae_params = getattr(ctx.actor, "get_vae_params", None)
        return callable(get_vae_params) and get_vae_params() is not None

    def compute(self, ctx: AuxLossContext) -> torch.Tensor:
        mu, log_var = ctx.actor.get_vae_params()
        # Per-dimension KL: -0.5 * (1 + log_var - mu^2 - exp(log_var))
        kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
        if self.kl_clip > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.kl_clip)
        return kl_per_dim.mean()


class ImitationLoss(_RegressionAuxiliaryLoss):
    """Behavior-cloning loss: student mean action vs. teacher privileged action.

    Applicable to distillation algorithms (``teacher is not None``) whose rollout buffer stores
    ``privileged_actions``. Extracted from ``PPODistillation.update``.
    """

    name = "imitation"

    def is_applicable(self, ctx: AuxLossContext) -> bool:
        return ctx.teacher is not None and getattr(ctx.batch, "privileged_actions", None) is not None

    def compute(self, ctx: AuxLossContext) -> torch.Tensor:
        # distribution_params[0] is the mean for a Gaussian distribution.
        student_actions_mean = ctx.distribution_params[0]
        return self.loss_fn(student_actions_mean, ctx.batch.privileged_actions)


class EncoderMatchingLoss(_RegressionAuxiliaryLoss):
    """Match the student encoder state to the teacher's privileged encoder state.

    Applicable when the rollout buffer holds both the student ``encoder_state`` and the teacher's
    ``privileged_encoder_state``. Extracted from ``PPODistillation.update`` /
    ``Distillation.update``.
    """

    name = "encoder_reconstruction"

    def is_applicable(self, ctx: AuxLossContext) -> bool:
        return (
            getattr(ctx.batch, "encoder_state", None) is not None
            and getattr(ctx.batch, "privileged_encoder_state", None) is not None
        )

    def compute(self, ctx: AuxLossContext) -> torch.Tensor:
        student_encoder_state = ctx.actor.get_encoder_state()
        return self.loss_fn(student_encoder_state, ctx.batch.privileged_encoder_state)


class DecoderMatchingLoss(_RegressionAuxiliaryLoss):
    """Match student and teacher latents *through the teacher's decoder*.

    Both the student encoder state and the teacher's privileged encoder state are passed through
    the *teacher's* decoder and the outputs are regressed onto each other. This is distinct from
    :class:`ReconstructionLoss` (which uses the actor's own decoder to rebuild its encoder obs).
    Extracted from ``PPODistillation.update`` / ``Distillation.update``.
    """

    name = "decoder_reconstruction"

    def is_applicable(self, ctx: AuxLossContext) -> bool:
        return (
            ctx.teacher is not None
            and getattr(ctx.batch, "encoder_state", None) is not None
            and getattr(ctx.batch, "privileged_encoder_state", None) is not None
        )

    def compute(self, ctx: AuxLossContext) -> torch.Tensor:
        student_encoder_state = ctx.actor.get_encoder_state()
        with torch.no_grad():
            teacher_decoder_output = ctx.teacher.get_decoder_inference(ctx.batch.privileged_encoder_state)
        student_decoder_output = ctx.teacher.get_decoder_inference(student_encoder_state)
        return self.loss_fn(student_decoder_output, teacher_decoder_output)


# ----------------------------------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------------------------------


# Loss kwargs that used to live directly on the algorithm config, now replaced by `aux_losses`.
_LEGACY_LOSS_KWARGS = (
    "decoder_loss_coef",
    "kl_loss_coef",
    "kl_clip",
    "imitation_loss_coef",
    "encoder_loss_coef",
    "loss_type",
    "loss_schedule",
    "total_iteration",
)


def reject_legacy_loss_kwargs(alg_cfg: dict) -> None:
    """Fail fast if an algorithm config still uses removed per-loss kwargs.

    These were replaced by the composable :func:`resolve_aux_losses` mechanism. We raise rather
    than silently translate, so a stale config errors loudly instead of training with a dropped
    loss term.

    Args:
        alg_cfg: Algorithm configuration dict.

    Raises:
        ValueError: If any removed loss kwarg is present, with migration guidance.
    """
    found = [key for key in _LEGACY_LOSS_KWARGS if key in alg_cfg]
    if found:
        raise ValueError(
            f"Algorithm config contains removed loss kwargs {found}. These were replaced by the "
            "composable `aux_losses` list (see rsl_rl/algorithms/aux_loss_design.md). For example, "
            "`decoder_loss_coef=c` becomes "
            "`aux_losses=[{'class_name': 'rsl_rl.algorithms.losses.ReconstructionLoss', 'coef': c}]`."
        )


def resolve_aux_losses(alg_cfg: dict) -> list[AuxiliaryLoss]:
    """Pop ``aux_losses`` from an algorithm config and instantiate the loss objects.

    Mirrors :func:`~rsl_rl.extensions.resolve_rnd_config` /
    :func:`~rsl_rl.extensions.resolve_symmetry_config`. Each spec is a dict with a ``class_name``
    (resolved via :func:`~rsl_rl.utils.resolve_callable`) and the remaining keys forwarded as
    keyword arguments to that loss class.

    Args:
        alg_cfg: Algorithm configuration dict. The ``aux_losses`` key (if present) is consumed.

    Returns:
        List of instantiated :class:`AuxiliaryLoss` objects (empty if none configured).
    """
    specs = alg_cfg.pop("aux_losses", None) or []
    losses: list[AuxiliaryLoss] = []
    for spec in specs:
        spec = dict(spec)  # don't mutate the caller's dict
        cls = resolve_callable(spec.pop("class_name"))
        losses.append(cls(**spec))
    return losses
