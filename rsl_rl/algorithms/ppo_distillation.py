# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.losses import reject_legacy_loss_kwargs, resolve_aux_losses
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class PPODistillation(PPO):
    r"""PPO with imitation losses from a frozen teacher policy.

    Trains the student (actor) jointly with the on-policy PPO objective (surrogate + value +
    entropy, using returns/advantages from the student's own rollouts) and a set of **auxiliary
    losses** against the frozen teacher. This class only adds the distillation-specific *rollout*
    machinery — recording the teacher's privileged actions and encoder state each step, freezing
    and eval-ing the teacher, and saving/loading it. The loss terms themselves are composable
    :class:`~rsl_rl.algorithms.losses.AuxiliaryLoss` objects supplied via ``aux_losses`` and run
    by the shared :meth:`PPO.update` loop. Typical distillation losses are:

    - :class:`~rsl_rl.algorithms.losses.ImitationLoss` — student mean action vs. teacher
      privileged action (behavior cloning).
    - :class:`~rsl_rl.algorithms.losses.EncoderMatchingLoss` — student encoder state vs. teacher
      encoder state (when both expose encoder states).
    - :class:`~rsl_rl.algorithms.losses.DecoderMatchingLoss` — student/teacher latents matched
      through the teacher's decoder (when the teacher has a decoder).

    The optional PPO/distillation curriculum is expressed via schedules: a decaying
    ``weight_schedule`` on the imitation loss together with the algorithm's ``ppo_weight_schedule``
    (see :meth:`PPO.update`). The teacher is always kept in eval mode with all parameters frozen.
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
        multi_gpu_cfg: dict | None = None,
        # Auxiliary losses
        aux_losses: list | None = None,
        ppo_weight_schedule: dict | None = None,
    ) -> None:
        """Initialize PPODistillation.

        Args:
            actor: Student model that interacts with the environment.
            teacher: Frozen privileged model used as imitation target. Parameters are
                immediately frozen (``requires_grad = False``).
            critic: Critic model for value function estimation.
            storage: Rollout storage (must be of type ``"ppo_distillation"``).
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
            multi_gpu_cfg: Optional multi-GPU configuration dict.
            aux_losses: Auxiliary loss terms (e.g. imitation / encoder / decoder matching) added
                to the PPO objective by :meth:`PPO.update`. Each self-guards on the model backend.
            ppo_weight_schedule: Optional schedule on the PPO term (the ``w_ppo`` half of a
                PPO/distillation curriculum). See :meth:`PPO.update`.
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
            multi_gpu_cfg=multi_gpu_cfg,
            aux_losses=aux_losses,
            ppo_weight_schedule=ppo_weight_schedule,
        )

        self.teacher = teacher.to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample student actions and record teacher privileged actions."""
        actions = super().act(obs)
        # Teacher inference (no grad, deterministic)
        self.transition.privileged_actions = self.teacher(obs).detach()
        teacher_enc = self.teacher.get_encoder_state()
        self.transition.privileged_encoder_state = teacher_enc.detach() if teacher_enc is not None else None
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record environment step and reset teacher recurrent state on episode ends."""
        super().process_env_step(obs, rewards, dones, extras)
        self.teacher.reset(dones)

    def train_mode(self) -> None:
        """Set student and critic to train mode; keep teacher in eval mode."""
        super().train_mode()
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set all models to eval mode."""
        super().eval_mode()
        self.teacher.eval()

    # NOTE: DAgger PPO without copying teacher actor weight to student actor
    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved = super().save()
        saved["teacher_state_dict"] = self.teacher.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None and any("actor_state_dict" in k for k in loaded_dict):
            # Loading from a PPO / privileged-policy checkpoint: only populate teacher
            load_cfg = {"teacher": True, "iteration": False}  # Only load teacher by default
        elif load_cfg is None: # Load from distillation training (inference)
            load_cfg = {
                "actor": True,
                "critic": True,
                "teacher": True,
                "optimizer": True,
                "iteration": True,
            }
        load_iteration = super().load(loaded_dict, load_cfg, strict)

        if load_cfg.get("teacher"):
            self.teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"], strict=strict
            )
            self.teacher_loaded = True

        return load_iteration

    # # NOTE: DAgger PPO copying teacher actor weight to student actor
    # # This is only valid when teacher/student actor/critic have same architecture
    # def save(self) -> dict:
    #     """Return a dict of all models for saving."""
    #     saved_dict = {
    #         "student_actor_state_dict": self.actor.state_dict(),
    #         "student_critic_state_dict": self.critic.state_dict(),
    #         "student_optimizer_state_dict": self.optimizer.state_dict(),
    #         "teacher_state_dict": self.teacher.state_dict(),
    #     }
    #     if self.rnd:
    #         saved_dict["rnd_state_dict"] = self.rnd.state_dict()
    #         saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
    #     return saved_dict

    # def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    #     """Load specified models from a saved dict."""
    #     if load_cfg is None and "actor_state_dict" in loaded_dict:
    #         # Loading from a PPO / privileged-policy checkpoint: populate teacher and
    #         # warm-start student actor MLP + critic from teacher weights.
    #         load_cfg = {"teacher": True, "init_student_from_teacher": True, "iteration": False}
    #     elif load_cfg is None:
    #         # Loading from a PPO distillation checkpoint (resume training or inference).
    #         load_cfg = {
    #             "actor": True,
    #             "critic": True,
    #             "teacher": True,
    #             "optimizer": True,
    #             "iteration": True,
    #         }

    #     # Load the specified models
    #     if load_cfg.get("actor"):
    #         self.actor.load_state_dict(loaded_dict["student_actor_state_dict"], strict=strict)
    #     if load_cfg.get("critic"):
    #         self.critic.load_state_dict(loaded_dict["student_critic_state_dict"], strict=strict)
    #     if load_cfg.get("optimizer"):
    #         self.optimizer.load_state_dict(loaded_dict["student_optimizer_state_dict"])
    #     if load_cfg.get("rnd") and self.rnd:
    #         self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
    #         self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    #     teacher_dict = loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"]
    #     if load_cfg.get("teacher"):
    #         self.teacher.load_state_dict(teacher_dict, strict=strict)
    #         self.teacher_loaded = True
    #     if load_cfg.get("init_student_from_teacher"):
    #         # Copy shared MLP weights (mlp, distribution) from teacher to student actor.
    #         # The encoder is excluded — different architecture (TCN vs VAE), learned from scratch.
    #         student_dict = self.actor.state_dict()
    #         for key, value in teacher_dict.items():
    #             if key.startswith("mlp.") or key.startswith("distribution."):
    #                 if key in student_dict and student_dict[key].shape == value.shape:
    #                     student_dict[key] = value
    #         self.actor.load_state_dict(student_dict, strict=False)
    #         # Warm-start critic from teacher checkpoint if available
    #         critic_src = loaded_dict.get("critic_state_dict")
    #         if critic_src is not None:
    #             self.critic.load_state_dict(critic_src, strict=strict)

    #     return load_cfg.get("iteration", False)

    def get_teacher(self) -> MLPModel:
        """Return the teacher model."""
        return self.teacher

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs (teacher included for consistent init)."""
        super().broadcast_parameters()
        model_params = [self.teacher.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.teacher.load_state_dict(model_params[0])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Reject removed per-loss kwargs, then resolve auxiliary losses (specs -> instances)
        reject_legacy_loss_kwargs(cfg["algorithm"])
        cfg["algorithm"]["aux_losses"] = resolve_aux_losses(cfg["algorithm"])

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

        storage = RolloutStorage(
            "ppo_distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        alg: PPODistillation = alg_class(
            student, teacher, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )

        return alg
