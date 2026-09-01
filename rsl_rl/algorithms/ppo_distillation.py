# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Generator
from tensordict import TensorDict
from typing import Callable

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, split_and_pad_trajectories


def _build_weight_scheduler(initial_value: float, schedule: dict | None) -> Callable[[int], float]:
    """Build a coefficient schedule callable ``step -> value``.

    Mirrors the constant/step/linear weight schedules used by RND (see
    :mod:`rsl_rl.extensions.rnd`), standalone so it can drive the PPO-term / distillation-term
    weights of a PPO/distillation curriculum without depending on any auxiliary-loss framework.

    Args:
        initial_value: Value returned when ``schedule`` is ``None`` (constant weight), or the
            default starting value if the schedule dict does not override it via ``"initial_value"``.
        schedule: Schedule config dict with a ``"mode"`` key, or ``None`` for a constant value.
            Supported modes:

            - ``"constant"``: always ``initial_value``.
            - ``"step"``: ``initial_value`` until ``final_step``, then ``final_value``.
            - ``"linear"``: linearly interpolate from ``initial_value`` (at ``initial_step``) to
              ``final_value`` (at ``final_step``).

    Returns:
        A callable mapping the current iteration to the scheduled weight.
    """
    if schedule is None:
        return lambda step: initial_value

    mode = schedule["mode"]
    initial_value = schedule.get("initial_value", initial_value)

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


class PPODistillationRolloutStorage(RolloutStorage):
    """On-policy ("rl") rollout storage augmented with the teacher's privileged actions.

    ``PPODistillation`` needs both the on-policy PPO buffers (values / log-probs / distribution
    params / returns / advantages) and the frozen teacher's action at every step (for the
    imitation loss) — neither the ``"rl"`` nor the ``"distillation"`` storage mode alone carries
    both, so this subclass adds the ``privileged_actions`` buffer on top of ``"rl"`` storage
    without touching :class:`~rsl_rl.storage.RolloutStorage` itself.
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        """Allocate ``"rl"`` rollout buffers plus a privileged-action buffer of the same shape."""
        super().__init__("rl", num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Add one transition, including the teacher's privileged action for this step."""
        step = self.step  # captured before the parent call increments it
        super().add_transition(transition)
        self.privileged_actions[step].copy_(transition.privileged_actions)  # type: ignore

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[RolloutStorage.Batch, None, None]:
        """Yield shuffled flat mini-batches, each carrying the matching privileged actions."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore
        privileged_actions = self.privileged_actions.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    privileged_actions=privileged_actions[batch_idx],
                )

    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Yield trajectory mini-batches (masks + hidden states), each with privileged actions."""
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        mini_batch_size = self.num_envs // num_mini_batches

        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                last_was_done = last_was_done.permute(1, 0)
                if self.saved_hidden_state_a is not None:
                    hidden_state_a_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_a
                    ]
                    hidden_state_a_batch = (
                        hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                    )
                else:
                    hidden_state_a_batch = None
                if self.saved_hidden_state_c is not None:
                    hidden_state_c_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_c
                    ]
                    hidden_state_c_batch = (
                        hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch
                    )
                else:
                    hidden_state_c_batch = None

                yield RolloutStorage.Batch(
                    observations=padded_obs_trajectories[:, first_traj:last_traj],  # type: ignore
                    actions=self.actions[:, start:stop],
                    values=self.values[:, start:stop],
                    advantages=self.advantages[:, start:stop],
                    returns=self.returns[:, start:stop],
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore
                    privileged_actions=self.privileged_actions[:, start:stop],
                    hidden_states=(hidden_state_a_batch, hidden_state_c_batch),  # type: ignore
                    masks=trajectory_masks[:, first_traj:last_traj],
                )

                first_traj = last_traj


class PPODistillation(PPO):
    r"""PPO with a behavior-cloning loss against a frozen teacher policy.

    Trains the student (actor) on the on-policy PPO objective (surrogate + value + entropy, using
    returns/advantages from the student's own rollouts) jointly with an imitation loss that
    regresses the student's mean action onto the teacher's (frozen, privileged) action at every
    step. Unlike upstream's composable auxiliary-loss framework, the distillation loss is computed
    directly inside :meth:`update`, and its weight relative to the PPO objective is driven by two
    independent schedules (see ``ppo_weight_schedule`` / ``distillation_weight_schedule``), enabling
    a PPO/distillation curriculum (e.g. behavior-cloning-heavy early on, annealing to pure PPO).

    The teacher is always kept in eval mode with all parameters frozen.
    """

    teacher: MLPModel
    """The frozen teacher (privileged) model."""

    teacher_loaded: bool = False
    """Indicates whether the teacher model parameters have been loaded from a checkpoint."""

    def __init__(
        self,
        actor: MLPModel,
        teacher: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # PPO parameters
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        # Aux-losses extension (Persona compositional networks)
        aux: object | None = None,
        multi_gpu_cfg: dict | None = None,
        # Distillation parameters
        distillation_loss_coef: float = 1.0,
        distillation_loss_type: str = "mse",
        ppo_weight_schedule: dict | None = None,
        distillation_weight_schedule: dict | None = None,
    ) -> None:
        """Initialize PPODistillation.

        Args:
            actor: Student model that interacts with the environment.
            teacher: Frozen privileged model used as the imitation target. Parameters are
                immediately frozen (``requires_grad = False``).
            critic: Critic model for value function estimation.
            storage: Rollout storage; must be a :class:`PPODistillationRolloutStorage`.
            num_learning_epochs: Number of PPO update epochs per rollout.
            num_mini_batches: Number of mini-batches per epoch.
            clip_param: PPO clipping parameter.
            gamma: Discount factor.
            lam: GAE lambda.
            value_loss_coef: Weight of the value function loss.
            entropy_coef: Weight of the entropy bonus.
            learning_rate: Initial learning rate.
            max_grad_norm: Maximum gradient norm for clipping.
            optimizer: Optimizer name (``"adam"`` or ``"sgd"``).
            use_clipped_value_loss: Whether to use the clipped value loss.
            schedule: Learning rate schedule (``"adaptive"`` or ``"fixed"``).
            desired_kl: Target KL divergence for adaptive LR scheduling.
            normalize_advantage_per_mini_batch: Normalize advantages within each mini-batch.
            device: Torch device string.
            rnd_cfg: Optional Random Network Distillation configuration dict.
            symmetry_cfg: Optional symmetry augmentation configuration dict.
            aux: Optional aux-losses extension (see :class:`~rsl_rl.algorithms.ppo.PPO`).
            multi_gpu_cfg: Optional multi-GPU configuration dict.
            distillation_loss_coef: Base weight of the imitation (behavior-cloning) loss; the
                starting value for ``distillation_weight_schedule`` when one is given.
            distillation_loss_type: Regression loss for the imitation term (``"mse"`` or ``"huber"``).
            ppo_weight_schedule: Optional schedule (see :func:`_build_weight_scheduler`) on the
                weight of the PPO term (surrogate + value + entropy). Defaults to a constant ``1.0``.
            distillation_weight_schedule: Optional schedule on the weight of the distillation term.
                Defaults to a constant ``distillation_loss_coef``.
        """
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            aux=aux,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.teacher = teacher.to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        # Distillation loss
        loss_fn_dict = {"mse": nn.functional.mse_loss, "huber": nn.functional.huber_loss}
        if distillation_loss_type not in loss_fn_dict:
            raise ValueError(
                f"Unknown distillation_loss_type: {distillation_loss_type}. Supported types are:"
                f" {list(loss_fn_dict.keys())}"
            )
        self.distillation_loss_fn = loss_fn_dict[distillation_loss_type]

        # PPO-term / distillation-term weight schedules (the PPO/distillation curriculum)
        self._ppo_weight_scheduler = _build_weight_scheduler(1.0, ppo_weight_schedule)
        self._distillation_weight_scheduler = _build_weight_scheduler(
            distillation_loss_coef, distillation_weight_schedule
        )
        self.current_iteration = 0


    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample student actions and record the teacher's privileged action."""
        actions = super().act(obs)
        self.transition.privileged_actions = self.teacher(obs).detach()
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record environment step and reset the teacher's recurrent state on episode ends."""
        super().process_env_step(obs, rewards, dones, extras)
        self.teacher.reset(dones)

    def update(self) -> dict[str, float]:
        """Run PPO update epochs jointly with the teacher-imitation loss, and return mean losses."""
        self.current_iteration += 1
        w_ppo = self._ppo_weight_scheduler(self.current_iteration)
        w_distillation = self._distillation_weight_scheduler(self.current_iteration)

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_distillation_loss = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # Aux losses (Persona): keyed by loss-term name, lazily populated
        mean_aux_losses: dict[str, float] = {}

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]
            # The policy's mean action: the distribution-agnostic imitation target (equals the
            # deterministic forward for every distribution, unlike distribution_params[0], which is
            # only the action mean for a Gaussian).
            student_action_mean = self.actor.output_mean[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Distillation (behavior-cloning) loss: student mean action vs. teacher privileged action
            distillation_loss = self.distillation_loss_fn(student_action_mean, batch.privileged_actions)

            # PPO/distillation curriculum: independently weight both terms
            loss = w_ppo * ppo_loss + w_distillation * distillation_loss

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Aux joint losses (Persona): summed into the main loss so shared parameters
            # (e.g. an encoder feeding both student and an aux head) receive gradients
            # from both objectives in one backward pass.
            if self.aux is not None:
                for aux_name, aux_loss in self.aux.compute_joint_losses(batch, original_batch_size).items():
                    loss = loss + aux_loss
                    mean_aux_losses[aux_name] = mean_aux_losses.get(aux_name, 0.0) + aux_loss.item()

            # Compute the gradients for PPO + distillation
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO + distillation
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()
            # Aux separate-optimizer losses (Persona): own forward/backward/step per term
            # (RND pattern); the extension all-reduces its own gradients when multi-GPU.
            if self.aux is not None:
                for aux_name, aux_value in self.aux.step_separate_losses(batch, original_batch_size).items():
                    mean_aux_losses[aux_name] = mean_aux_losses.get(aux_name, 0.0) + aux_value

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_distillation_loss += distillation_loss.item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_distillation_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "distillation": mean_distillation_loss,
            "weight_ppo": w_ppo,
            "weight_distillation": w_distillation,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        for aux_name, aux_total in mean_aux_losses.items():
            loss_dict[aux_name] = aux_total / num_updates

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set student and critic to train mode; keep the teacher in eval mode."""
        super().train_mode()
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set all models to eval mode."""
        super().eval_mode()
        self.teacher.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved = super().save()
        saved["teacher_state_dict"] = self.teacher.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None and any("teacher_state_dict" in k for k in loaded_dict):
            # Loading from a PPO distillation checkpoint (resume training or inference)
            load_cfg = {
                "actor": True,
                "critic": True,
                "teacher": True,
                "optimizer": True,
                "iteration": True,
                "aux": True,
            }
        elif load_cfg is None and any("actor_state_dict" in k for k in loaded_dict):
            # Loading from a PPO / privileged-policy checkpoint: only populate the teacher
            load_cfg = {"teacher": True, "iteration": False}
        load_iteration = super().load(loaded_dict, load_cfg, strict)

        if load_cfg.get("teacher"):
            self.teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"], strict=strict
            )
            self.teacher_loaded = True

        return load_iteration

    def get_teacher(self) -> MLPModel:
        """Return the teacher model."""
        return self.teacher

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPODistillation:
        """Construct the PPODistillation algorithm from a config dict.

        Expected top-level keys in ``cfg``:
        - ``"algorithm"``: algorithm hyperparameters (including ``class_name``).
        - ``"student"``: student model config (including ``class_name``).
        - ``"teacher"``: teacher model config (including ``class_name``).
        - ``"critic"``: critic model config (including ``class_name``).
        - ``"obs_groups"``: observation group mapping.
        - ``"num_steps_per_env"``: rollout length per environment.
        - ``"multi_gpu"``: multi-GPU config or ``None``.
        """
        alg_class: type[PPODistillation] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["student", "teacher", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(
            device
        )
        print(f"Student Model: {student}")

        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(
            device
        )
        print(f"Teacher Model: {teacher}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = student.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = PPODistillationRolloutStorage(
            env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        alg: PPODistillation = alg_class(
            student, teacher, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs (teacher included for consistent init)."""
        super().broadcast_parameters()
        model_params = [self.teacher.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.teacher.load_state_dict(model_params[0])