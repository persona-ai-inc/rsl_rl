# Algorithms

This directory contains the learning algorithms available in this fork of RSL-RL. All
algorithms share a common interface (`act`, `process_env_step`, `compute_returns`, `update`,
`save`, `load`, `construct_algorithm`, ...) so they can be driven by the same runner. They are
registered in [`__init__.py`](__init__.py).

| Algorithm | Class | Base | Models | Objective |
|-----------|-------|------|--------|-----------|
| PPO | `PPO` | — | actor, critic | RL (surrogate + value + entropy) + optional auxiliary losses |
| Distillation | `Distillation` | — | student, teacher | Behavior cloning |
| PPO + Distillation | `PPODistillation` | `PPO` | student (actor), teacher, critic | PPO + imitation (auxiliary losses) |

`PPODistillation` derives from `PPO`; `Distillation` is standalone.

> **Auto-encoder / VAE actors.** There are no longer separate `PPOAE` / `PPOVAE` classes. An
> actor with a decoder (`MLPAutoEncoderModel`) or a variational encoder (`MLPVAEModel`) is trained
> with plain `PPO` plus the appropriate entries in the **[auxiliary loss](#auxiliary-losses)**
> list (`ReconstructionLoss`, `VAEKLLoss`). See the design notes in
> [`aux_loss_design.md`](aux_loss_design.md).

---

## PPO

[`ppo.py`](ppo.py) — Proximal Policy Optimization
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)).

The base on-policy RL algorithm. Collects rollouts with a stochastic actor, computes returns
and advantages with Generalized Advantage Estimation (GAE), and updates the actor and critic
over several epochs of mini-batches.

**Loss:**

```
L = ppo_weight * (L_surrogate + value_loss_coef * L_value - entropy_coef * H)
  + sum_i  coef_i * L_aux_i        # optional auxiliary losses
```

- **Surrogate loss** — clipped PPO policy-gradient objective with `clip_param`.
- **Value loss** — (optionally clipped) MSE between predicted values and GAE returns.
- **Entropy bonus** — encourages exploration.
- **Auxiliary losses** — any number of composable, self-guarding extra terms (see below);
  empty by default, in which case the objective is exactly standard PPO.

**Key features:**

- Adaptive learning-rate scheduling based on a target KL divergence (`desired_kl`).
- Time-out bootstrapping for truncated episodes.
- Optional extensions: **Random Network Distillation (RND)** for intrinsic rewards and
  **Symmetry** augmentation / mirror loss.
- **Auxiliary losses** via `aux_losses`, with an optional `ppo_weight_schedule` on the PPO term
  (used together with a decaying imitation schedule to form a distillation curriculum).
- Supports recurrent policies, multi-GPU (distributed) training, and `torch.compile`.

---

## Auxiliary Losses

[`losses.py`](losses.py) — composable, self-describing training-loss terms added to the PPO
objective during `update()`.

Each loss is an `AuxiliaryLoss` that:

- has a **name** (its key in the logged loss dict),
- **self-guards** via `is_applicable(ctx)` — it skips silently when the model backend lacks the
  hook it needs (e.g. a reconstruction loss on a plain-MLP actor),
- computes a **raw, unweighted** scalar via `compute(ctx)`,
- exposes a **coefficient**, optionally driven by a weight schedule.

Losses are selected **explicitly by config** (a list of specs resolved by `resolve_aux_losses`)
*and* self-guard, so the same list can be reused across MLP / AutoEncoder / VAE / RNN / TCN
backends — inapplicable terms become no-ops rather than errors.

| Class | name (loss-dict key) | Applicable when | Raw loss |
|-------|----------------------|-----------------|----------|
| `ReconstructionLoss` | `decoder` | actor has a decoder (`has_decoder`) | `loss_fn(decoder_output, encoder_obs)` |
| `VAEKLLoss` | `kl` | actor exposes `get_vae_params()` | `mean( max(kl_clip, -0.5·(1 + logσ² − μ² − σ²)) )` |
| `ImitationLoss` | `imitation` | distillation (teacher present) + `privileged_actions` | `loss_fn(student_mean, teacher_action)` |
| `EncoderMatchingLoss` | `encoder_reconstruction` | student + teacher encoder states stored | `loss_fn(student_enc, teacher_enc)` |
| `DecoderMatchingLoss` | `decoder_reconstruction` | encoder states stored (teacher present) | `loss_fn(teacher_dec(student_enc), teacher_dec(teacher_enc))` |

Common parameters: `coef` (weight), `loss_type` (`"mse"` / `"huber"`, regression losses only),
`weight_schedule` (optional `constant` / `step` / `linear` schedule). `VAEKLLoss` additionally
takes `kl_clip` (free-nats threshold; `0.0` disables clipping).

Note that `ReconstructionLoss` (the actor's own decoder rebuilds its encoder obs) is distinct
from `DecoderMatchingLoss` (student and teacher latents are both passed through the *teacher's*
decoder and matched).

---

## Distillation

[`distillation.py`](distillation.py) — Policy distillation / behavior cloning.

Trains a **student** model to mimic a frozen **teacher** model in a purely supervised fashion
(no RL objective, no return/advantage computation). The teacher's parameters are frozen on
construction. During rollouts the student acts in the environment while the teacher's privileged
actions (and optionally encoder states) are recorded; the student is then regressed onto those
targets.

**Loss (sum of available terms):**

```
L = L_behavior + L_encoder_reconstruction + L_decoder_reconstruction
```

- **Behavior loss** — student actions vs. teacher privileged actions.
- **Encoder reconstruction loss** — student encoder state vs. teacher encoder state (when both
  expose encoder states).
- **Decoder matching loss** — student vs. teacher decoder outputs, both passed through the
  teacher decoder.

**Key parameters / features:**

- `gradient_length` — number of accumulated steps between optimizer updates (truncated BPTT for
  recurrent students); `num_learning_epochs` controls epochs per rollout.
- `loss_type` — `"mse"` or `"huber"`.
- Supports loading a teacher directly from a PPO checkpoint (`actor_state_dict`).
- **Not** compatible with the RND or Symmetry extensions.

---

## PPO with Distillation (`PPODistillation`)

[`ppo_distillation.py`](ppo_distillation.py) — joint on-policy RL and imitation (DAgger-style
PPO).

Extends `PPO` to train a student (actor) **simultaneously** with the standard PPO RL objective
and imitation losses against a frozen teacher policy. The student collects its own rollouts
(so RL returns/advantages come from the student's own value function), while the teacher's
privileged actions and encoder states are recorded alongside for the imitation terms. The
teacher is always frozen and in eval mode.

This class only adds the distillation-specific *rollout* machinery (recording the teacher's
privileged actions / encoder state, freezing + eval-ing the teacher, saving/loading it). The
loss terms themselves are **[auxiliary losses](#auxiliary-losses)** run by the shared
`PPO.update` loop.

**Loss:**

```
L = ppo_weight * L_PPO
  + coef_imit * L_imit           # ImitationLoss
  + coef_enc  * L_enc            # EncoderMatchingLoss (optional)
  + coef_dec  * L_dec            # DecoderMatchingLoss (optional)
```

- **PPO loss** — standard surrogate + value + entropy on the student's rollouts.
- **`ImitationLoss`** — student mean action vs. teacher privileged action (behavior cloning).
- **`EncoderMatchingLoss`** (optional) — student vs. teacher encoder states.
- **`DecoderMatchingLoss`** (optional) — student/teacher latents matched through the teacher's
  decoder.

**Key parameters:**

- `aux_losses` — the imitation / encoder / decoder matching terms (each with its own `coef` and
  `loss_type`). Include only the terms you want; omitted terms are simply absent.
- `ppo_weight_schedule` — optional schedule on the PPO term. A PPO/distillation **curriculum** is
  expressed as a decaying `weight_schedule` on `ImitationLoss` together with this rising
  `ppo_weight_schedule`.
- Inherits PPO's RND, Symmetry, multi-GPU, and recurrent-policy support.

Uses a dedicated `"ppo_distillation"` rollout storage that holds both PPO transition data and
the teacher's privileged targets.



# Configuration Examples
To use custom-made NN architectures and algorithms, define config class and parse it to rsl_rl agent config files. 

## Model config
```python
# may defined in isaaclab_rl/isaaclab_rl/rsl_rl/rl_model_cfg.py

@configclass
class RslRlMLPEncoderModelCfg(RslRlMLPModelCfg):
    """Configuration for MLP Encoder model."""

    class_name: str = "MLPEncoderModel"
    """The model class name. Defaults to MLPEncoderModel."""
    encoder_obs_set: str = MISSING
    """The observation set for the encoder."""
    encoder_output_dim: int = MISSING
    """The output dimension of the encoder."""
    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder."""
    encoder_activation: str = MISSING
    """The activation function for the encoder."""
    encoder_obs_normalization: bool = False
    """Whether to use observation normalization for the encoder. Defaults to False."""


@configclass
class RslRlMLPAEModelCfg(RslRlMLPModelCfg):
    """Configuration for MLP AutoEncoder model."""

    class_name: str = "MLPAutoEncoderModel"
    """The model class name. Defaults to MLPAutoEncoderModel."""
    encoder_obs_set: str = MISSING
    """The observation set for the encoder."""
    encoder_output_dim: int = MISSING
    """The output dimension of the encoder."""
    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder."""
    encoder_activation: str = MISSING
    """The activation function for the encoder."""
    encoder_obs_normalization: bool = False
    """Whether to use observation normalization for the encoder. Defaults to False."""
    decoder_obs_set: str = MISSING
    """The observation set for the decoder."""


@configclass
class RslRlMLPVAEModelCfg(RslRlMLPModelCfg):
    """Configuration for MLP Variational AutoEncoder model."""

    class_name: str = "MLPVAEModel"
    """The model class name. Defaults to MLPVAEModel."""
    encoder_obs_set: str = MISSING
    """The observation set for the encoder."""
    encoder_output_dim: int = MISSING
    """The output dimension of the encoder."""
    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder."""
    encoder_activation: str = MISSING
    """The activation function for the encoder."""
    encoder_obs_normalization: bool = False
    """Whether to use observation normalization for the encoder. Defaults to False."""
    decoder_obs_set: str = MISSING
    """The observation set for the decoder."""


@configclass
class RslRlRNNEncoderModelCfg(RslRlMLPModelCfg):
    """Configuration for RNN Encoder model."""

    class_name: str = "RNNEncoderModel"
    """The model class name. Defaults to RNNEncoderModel."""
    encoder_obs_set: str = MISSING
    """The observation set for the encoder."""
    encoder_obs_normalization: bool = False
    """Whether to use observation normalization for the encoder. Defaults to False."""
    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""
    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""
    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlTCNModelCfg(RslRlMLPModelCfg):
    """Configuration for TCN model."""

    class_name: str = "TCNModel"
    """The model class name. Defaults to TCNModel."""
    encoder_obs_set: str = MISSING
    """The observation set for the encoder."""
    encoder_output_dim: int = MISSING
    """The output dimension of the encoder."""
    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder."""
    encoder_activation: str = MISSING
    """The activation function for the encoder."""
    encoder_obs_normalization: bool = False
    """Whether to use observation normalization for the encoder. Defaults to False."""


@configclass
class RslRlTCNAttentionModelCfg(RslRlMLPModelCfg):
    """Configuration for TCN Attention model."""

    class_name: str = "TCNAttentionModel"
    """The model class name. Defaults to TCNAttentionModel."""
    encoder_obs_set: str = MISSING
    """The observation set for the encoder."""
    encoder_output_dim: int = MISSING
    """The output dimension of the encoder."""
    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder."""
    encoder_activation: str = MISSING
    """The activation function for the encoder."""
    encoder_obs_normalization: bool = False
    """Whether to use observation normalization for the encoder. Defaults to False."""
```

## Loss config

Auxiliary losses are configured as a list on the algorithm config. Each entry is an `AuxLossCfg`
whose `class_name` points at one of the loss classes in [`losses.py`](losses.py); the remaining
fields are forwarded as constructor kwargs. There are **no** `PPOAE` / `PPOVAE` algorithm configs
— an AE/VAE actor uses the base `RslRlPpoAlgorithmCfg` plus the right `aux_losses`.

```python
# may defined in isaaclab_rl/isaaclab_rl/rsl_rl/rl_loss_cfg.py

@configclass
class AuxLossCfg:
    """Base configuration for an auxiliary loss term."""

    class_name: str = MISSING
    coef: float = 1.0
    weight_schedule: dict | None = None


@configclass
class RegressionAuxLossCfg(AuxLossCfg):
    """Auxiliary loss that regresses one tensor onto another."""

    loss_type: Literal["mse", "huber"] = "mse"


@configclass
class ReconstructionLossCfg(RegressionAuxLossCfg):
    class_name: str = "rsl_rl.algorithms.losses.ReconstructionLoss"


@configclass
class VAEKLLossCfg(AuxLossCfg):
    class_name: str = "rsl_rl.algorithms.losses.VAEKLLoss"
    kl_clip: float = 0.0


@configclass
class ImitationLossCfg(RegressionAuxLossCfg):
    class_name: str = "rsl_rl.algorithms.losses.ImitationLoss"


@configclass
class EncoderMatchingLossCfg(RegressionAuxLossCfg):
    class_name: str = "rsl_rl.algorithms.losses.EncoderMatchingLoss"


@configclass
class DecoderMatchingLossCfg(RegressionAuxLossCfg):
    class_name: str = "rsl_rl.algorithms.losses.DecoderMatchingLoss"
```

## Algorithm config

```python 
# may defined in isaaclab_rl/isaaclab_rl/rsl_rl/rl_alg_cfg.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    # RslRlPpoAlgorithmCfg,
    # RslRlRNNModelCfg,
    RslRlSymmetryCfg,
)
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg as BaseRslRlPpoAlgorithmCfg

# The base PPO algorithm config gains a single `aux_losses` field; `PPODistillation` reuses it.
@configclass
class RslRlPpoAlgorithmCfg(BaseRslRlPpoAlgorithmCfg):
    # additional to existing PPO parameters
    aux_losses: list[AuxLossCfg] = []
    ppo_weight_schedule: dict | None = None


@configclass
class RslRlPpoDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for PPO with teacher distillation."""

    class_name: str = "PPODistillation"
```


## Full config in agent directory 


### Example1: PPO with VAE actor and MLP critic
```python 
@configclass
class PPOVAERunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "encoder": ["encoder input"],
        "decoder": ["decoder output"],
    }

    actor = RslRlMLPVAEModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        encoder_obs_set="encoder",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
        encoder_obs_normalization=False,
        decoder_obs_set="decoder",
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    # VAE actor => plain PPO + reconstruction + KL auxiliary losses
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        aux_losses=[
            ReconstructionLossCfg(coef=1.0, loss_type="mse"),
            VAEKLLossCfg(coef=0.01, kl_clip=0.0),
        ],
    )
```

### Example2: PPO-DAgger with VAE teacher actor and TCN student actor

```python
@configclass
class PPODistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    obs_groups = {
        "teacher": ["policy"],
        "teacher_encoder": ["teacher_encoder_input"],
        "teacher_decoder": ["teacher_decoder_target"],
        "student": ["policy"],
        "critic": ["critic"],
        "student_encoder": ["student_encoder"],
    }

    teacher = RslRlMLPVAEModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        encoder_obs_set="teacher_encoder",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
        encoder_obs_normalization=False,
        decoder_obs_set="teacher_decoder",
    )

    # TCN student
    student = RslRlTCNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        encoder_obs_set="student_encoder",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
    )

    # student critic
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    algorithm = RslRlPpoDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        # num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        aux_losses=[
            # Curriculum: imitation weight decays as the PPO weight rises (see ppo_weight_schedule).
            # For a "fixed" schedule, simply drop the weight_schedule (constant coef=1.0).
            ImitationLossCfg(
                coef=1.0,
                loss_type="mse",
                weight_schedule={
                    "mode": "linear",
                    "initial_step": 0,
                    "final_step": max_iterations // 2,
                    "final_value": 0.1,
                },
            ),
            EncoderMatchingLossCfg(coef=1.0, loss_type="mse"),
            DecoderMatchingLossCfg(coef=1.0, loss_type="mse"),
        ],
        ppo_weight_schedule={
            "mode": "linear",
            "initial_value": 0.0,
            "initial_step": 0,
            "final_step": max_iterations // 2,
            "final_value": 0.9,
        },
    )

```

### Example3: PPO-DAgger with MLP teacher actor and MLP student actor

```python
@configclass
class PPODistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    obs_groups = {
        "teacher": ["teacher"],
        "student": ["policy"],
        "critic": ["critic"],
    }

    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )

    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    algorithm = RslRlPpoDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        aux_losses=[
            # Curriculum: imitation weight decays as the PPO weight rises (see ppo_weight_schedule).
            # For a "fixed" schedule, simply drop the weight_schedule (constant coef=1.0).
            ImitationLossCfg(
                coef=1.0,
                loss_type="mse",
                weight_schedule={
                    "mode": "linear",
                    "initial_step": 0,
                    "final_step": max_iterations // 2,
                    "final_value": 0.1,
                },
            ),
        ],
        ppo_weight_schedule={
            "mode": "linear",
            "initial_value": 0.0,
            "initial_step": 0,
            "final_step": max_iterations // 2,
            "final_value": 0.9,
        },
    )

```