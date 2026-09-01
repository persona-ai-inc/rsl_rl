# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from rsl_rl.algorithms.ppo_distillation import PPODistillation
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class PPODistillationRunner(OnPolicyRunner):
    """On-policy runner for :class:`~rsl_rl.algorithms.ppo_distillation.PPODistillation`.

    ``PPODistillation`` collects rollouts and updates exactly like on-policy PPO from the
    runner's point of view; only the algorithm class differs. This subclass adds no behavior over
    :class:`OnPolicyRunner` — it exists so training configs can pin the runner via ``class_name``
    alongside the algorithm.
    """

    alg: PPODistillation
    """The PPO distillation algorithm."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop after validating that the teacher model is loaded."""
        # Check if teacher is loaded
        if not self.alg.teacher_loaded:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        super().learn(num_learning_iterations, init_at_random_ep_len)
