# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PPODistillation algorithm."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_distillation import PPODistillation, PPODistillationRolloutStorage, _build_weight_scheduler
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_actor(obs: TensorDict, obs_groups: dict, obs_set: str, **kwargs: object) -> MLPModel:
    """Create an MLPModel actor/teacher with a Gaussian distribution."""
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, obs_set, NUM_ACTIONS, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    """Create an MLPModel critic (no distribution)."""
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **defaults)


def _build_ppo_distillation(**overrides: object) -> tuple[PPODistillation, TensorDict]:
    """Build a PPODistillation instance with small networks for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"student": ["policy"], "teacher": ["policy"], "critic": ["policy"]}
    student = _make_actor(obs, obs_groups, "student")
    teacher = _make_actor(obs, obs_groups, "teacher")
    critic = _make_critic(obs, obs_groups)
    storage = PPODistillationRolloutStorage(NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        schedule="fixed",
        desired_kl=0.01,
    )
    defaults.update(overrides)
    alg = PPODistillation(student, teacher, critic, storage, **defaults)
    return alg, obs


def _fill_and_return(alg: PPODistillation, obs: TensorDict) -> None:
    """Roll out NUM_STEPS transitions and compute returns/advantages."""
    for _ in range(NUM_STEPS):
        alg.act(obs)
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        alg.process_env_step(obs, rewards, dones, {})
    alg.compute_returns(obs)


class TestWeightScheduler:
    """Tests for the standalone PPO/distillation weight scheduler."""

    def test_constant_schedule(self) -> None:
        scheduler = _build_weight_scheduler(1.0, None)
        assert scheduler(0) == 1.0
        assert scheduler(1000) == 1.0

    def test_step_schedule(self) -> None:
        scheduler = _build_weight_scheduler(1.0, {"mode": "step", "final_step": 10, "final_value": 0.0})
        assert scheduler(0) == 1.0
        assert scheduler(9) == 1.0
        assert scheduler(10) == 0.0

    def test_linear_schedule(self) -> None:
        scheduler = _build_weight_scheduler(
            1.0, {"mode": "linear", "initial_step": 0, "final_step": 10, "final_value": 0.0}
        )
        assert scheduler(0) == 1.0
        assert scheduler(10) == 0.0
        assert abs(scheduler(5) - 0.5) < 1e-6


class TestPPODistillationRollout:
    """Tests for rollout collection (act / process_env_step)."""

    def test_act_records_privileged_actions(self) -> None:
        """Each transition should record the frozen teacher's action alongside the student's."""
        alg, obs = _build_ppo_distillation()
        actions = alg.act(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        assert alg.transition.privileged_actions is not None
        assert alg.transition.privileged_actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_storage_carries_privileged_actions_through_update(self) -> None:
        """Mini-batches produced during update() should carry matching privileged actions."""
        alg, obs = _build_ppo_distillation(num_learning_epochs=1, num_mini_batches=1)
        _fill_and_return(alg, obs)
        alg.train_mode()
        loss_dict = alg.update()
        assert "distillation" in loss_dict
        assert loss_dict["distillation"] >= 0.0


class TestPPODistillationUpdate:
    """Tests for the joint PPO + distillation update loop."""

    def test_update_changes_student_and_critic_but_not_teacher(self) -> None:
        """Student/critic parameters change after update; teacher stays frozen."""
        alg, obs = _build_ppo_distillation()
        alg.train_mode()

        student_before = {name: p.clone() for name, p in alg.actor.named_parameters()}
        critic_before = {name: p.clone() for name, p in alg.critic.named_parameters()}
        teacher_before = {name: p.clone() for name, p in alg.teacher.named_parameters()}

        _fill_and_return(alg, obs)
        alg.update()

        assert any(not torch.equal(p, student_before[name]) for name, p in alg.actor.named_parameters())
        assert any(not torch.equal(p, critic_before[name]) for name, p in alg.critic.named_parameters())
        for name, p in alg.teacher.named_parameters():
            assert torch.equal(p, teacher_before[name]), f"Teacher parameter {name} changed during update"

    def test_distillation_loss_decreases_with_pure_imitation(self) -> None:
        """With w_ppo=0, repeated updates should drive the student toward the teacher's actions."""
        alg, obs = _build_ppo_distillation(
            num_learning_epochs=4,
            num_mini_batches=1,
            ppo_weight_schedule={"mode": "constant", "initial_value": 0.0},
            distillation_loss_coef=1.0,
        )
        alg.train_mode()

        losses = []
        for _ in range(5):
            _fill_and_return(alg, obs)
            loss_dict = alg.update()
            losses.append(loss_dict["distillation"])

        assert losses[-1] < losses[0], f"Distillation loss should decrease, got {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_weight_schedule_reported_in_loss_dict(self) -> None:
        """The scheduled ppo/distillation weights for this iteration should be logged."""
        alg, obs = _build_ppo_distillation(
            num_learning_epochs=1,
            num_mini_batches=1,
            ppo_weight_schedule={"mode": "step", "final_step": 2, "final_value": 0.0},
        )
        alg.train_mode()

        _fill_and_return(alg, obs)
        loss_dict = alg.update()
        assert loss_dict["weight_ppo"] == 1.0  # current_iteration == 1 < final_step == 2

        _fill_and_return(alg, obs)
        loss_dict = alg.update()
        assert loss_dict["weight_ppo"] == 0.0  # current_iteration == 2 >= final_step == 2

    def test_recurrent_generator_not_used_for_feedforward_models(self) -> None:
        """Feedforward student/critic should use the flat mini-batch generator without error."""
        alg, obs = _build_ppo_distillation()
        assert not alg.actor.is_recurrent
        assert not alg.critic.is_recurrent
        _fill_and_return(alg, obs)
        alg.train_mode()
        loss_dict = alg.update()
        for key in ("value", "surrogate", "entropy", "distillation"):
            assert key in loss_dict


class TestPPODistillationSaveLoad:
    """Tests for checkpoint save/load semantics."""

    def test_save_includes_teacher(self) -> None:
        alg, _obs = _build_ppo_distillation()
        saved = alg.save()
        assert "teacher_state_dict" in saved
        assert "actor_state_dict" in saved
        assert "critic_state_dict" in saved

    def test_load_from_ppo_checkpoint_only_populates_teacher(self) -> None:
        """Loading a plain PPO checkpoint (actor_state_dict, no teacher) should only seed the teacher."""
        alg, _obs = _build_ppo_distillation()
        source_actor_state = {k: v.clone() for k, v in alg.actor.state_dict().items()}
        ppo_checkpoint = {"actor_state_dict": source_actor_state}

        assert not alg.teacher_loaded
        load_iteration = alg.load(ppo_checkpoint, load_cfg=None, strict=True)
        assert alg.teacher_loaded
        assert load_iteration is False
        for key, value in source_actor_state.items():
            assert torch.equal(alg.teacher.state_dict()[key], value)
