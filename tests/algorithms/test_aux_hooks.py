# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the aux-losses extension hooks on PPO."""

from __future__ import annotations

import torch
from collections.abc import Iterable
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4
NUM_EPOCHS = 2
NUM_MINI_BATCHES = 2


class _RecordingAux:
    """Aux extension that records the iteration passed to each hook."""

    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.joint_iterations: list[int] = []
        self.separate_iterations: list[int] = []

    def joint_parameters(self) -> Iterable[torch.nn.Parameter]:
        yield self.param

    def compute_joint_losses(self, batch: object, original_batch_size: int, iteration: int) -> dict[str, torch.Tensor]:
        self.joint_iterations.append(iteration)
        return {"aux_joint": self.param.sum() ** 2}

    def step_separate_losses(self, batch: object, original_batch_size: int, iteration: int) -> dict[str, float]:
        self.separate_iterations.append(iteration)
        return {"aux_separate": 0.5}

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


def _build_ppo(aux: _RecordingAux) -> tuple[PPO, TensorDict]:
    """Build a small PPO instance wired to the given aux extension."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[32, 32],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[32, 32], activation="elu")
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    ppo = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=NUM_EPOCHS,
        num_mini_batches=NUM_MINI_BATCHES,
        learning_rate=1e-3,
        schedule="fixed",
        aux=aux,
    )
    return ppo, obs


def _fill_and_return(ppo: PPO, obs: TensorDict) -> None:
    """Roll out NUM_STEPS transitions and compute returns/advantages."""
    for _ in range(NUM_STEPS):
        ppo.act(obs)
        ppo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
    ppo.compute_returns(obs)


class TestAuxIterationHook:
    """The runner's learning iteration must reach both aux hooks."""

    def test_iteration_forwarded_to_both_hooks(self) -> None:
        """Every mini-batch call in one update sees the iteration the runner passed."""
        aux = _RecordingAux()
        ppo, obs = _build_ppo(aux)
        _fill_and_return(ppo, obs)
        ppo.update(7)

        num_updates = NUM_EPOCHS * NUM_MINI_BATCHES
        assert aux.joint_iterations == [7] * num_updates
        assert aux.separate_iterations == [7] * num_updates

    def test_iteration_defaults_to_zero(self) -> None:
        """``update()`` without an iteration still works, for callers outside the runner."""
        aux = _RecordingAux()
        ppo, obs = _build_ppo(aux)
        _fill_and_return(ppo, obs)
        ppo.update()

        assert set(aux.joint_iterations) == {0}
        assert set(aux.separate_iterations) == {0}
