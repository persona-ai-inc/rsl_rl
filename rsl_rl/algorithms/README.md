# Algorithms

This directory contains the learning algorithms available in this fork of RSL-RL. All
algorithms share a common interface (`act`, `process_env_step`, `compute_returns`, `update`,
`save`, `load`, `construct_algorithm`, ...) so they can be driven by the same runner. They are
registered in [`__init__.py`](__init__.py).

| Algorithm | Class | Base | Models | Objective |
|-----------|-------|------|--------|-----------|
| PPO | `PPO` | — | actor, critic | RL (surrogate + value + entropy) |
| PPO + Auto-Encoder | `PPOAE` | `PPO` | actor (w/ decoder), critic | PPO + reconstruction |
| PPO + VAE | `PPOVAE` | `PPOAE` | actor (w/ VAE), critic | PPO + reconstruction + latent KL |
| Distillation | `Distillation` | — | student, teacher | Behavior cloning |
| PPO + Distillation | `PPODistillation` | `PPO` | student (actor), teacher, critic | PPO + imitation |

The class hierarchy is: `PPO → PPOAE → PPOVAE`, with `PPODistillation` also deriving from `PPO`.
`Distillation` is standalone.

---

## PPO

[`ppo.py`](ppo.py) — Proximal Policy Optimization
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)).

The base on-policy RL algorithm. Collects rollouts with a stochastic actor, computes returns
and advantages with Generalized Advantage Estimation (GAE), and updates the actor and critic
over several epochs of mini-batches.

**Loss:**

```
L = L_surrogate + value_loss_coef * L_value - entropy_coef * H
```

- **Surrogate loss** — clipped PPO policy-gradient objective with `clip_param`.
- **Value loss** — (optionally clipped) MSE between predicted values and GAE returns.
- **Entropy bonus** — encourages exploration.

**Key features:**

- Adaptive learning-rate scheduling based on a target KL divergence (`desired_kl`).
- Time-out bootstrapping for truncated episodes.
- Optional extensions: **Random Network Distillation (RND)** for intrinsic rewards and
  **Symmetry** augmentation / mirror loss.
- Supports recurrent policies, multi-GPU (distributed) training, and `torch.compile`.

---

## PPO with Auto-Encoder (`PPOAE`)

[`ppo_ae.py`](ppo_ae.py) — PPO with an auxiliary reconstruction objective.

Extends `PPO` for actors that carry a decoder (`actor.has_decoder is True`, e.g.
`MLPAutoEncoderModel`). In addition to the PPO loss, the decoder is trained to reconstruct the
encoder's input observations from the encoder bottleneck latent. This regularizes the learned
latent representation.

**Loss:**

```
L = L_PPO + decoder_loss_coef * L_recon
```

where `L_recon = loss_fn(decoder_output, encoder_observations)`. The reconstruction loss is
**added to the PPO loss**, so a single optimizer step jointly updates the encoder/decoder and the
policy MLP.

**Key parameters:**

- `decoder_loss_coef` — weight of the reconstruction loss.
- `loss_type` — regression loss for reconstruction (`"mse"` or `"huber"`).

If the actor has no decoder, `update()` raises an error.

---

## PPO with VAE (`PPOVAE`)

[`ppo_vae.py`](ppo_vae.py) — PPO with a β-VAE structured latent.

Extends `PPOAE` by treating the encoder as a variational encoder and adding a KL-divergence term
between the encoder posterior `q(z|x)` and an isotropic Gaussian prior `N(0, I)`. Designed for
actors that expose `get_vae_params()` returning `(mu, log_var)` (e.g. `MLPVAEModel`).

**Loss:**

```
L = L_PPO + decoder_loss_coef * L_recon + kl_loss_coef * L_KL
```

where the per-dimension KL is

```
L_KL = mean( max(kl_clip, -0.5 * (1 + log_var - mu^2 - exp(log_var))) )
```

**Key parameters:**

- `kl_loss_coef` — weight of the VAE KL loss.
- `kl_clip` — optional *free-nats* threshold; only KL above this per-dimension tolerance is
  penalized, which prevents posterior collapse. Set to `0.0` for a standard VAE.
- Inherits `decoder_loss_coef` and `loss_type` from `PPOAE`.

If the actor has a decoder but no `get_vae_params()`, it falls back to plain `PPOAE` behavior
(no KL term).

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
and an imitation loss against a frozen teacher policy. The student collects its own rollouts
(so RL returns/advantages come from the student's own value function), while the teacher's
privileged actions and encoder states are recorded alongside for the imitation terms. The
teacher is always frozen and in eval mode.

**Loss:**

```
L = w_ppo * L_PPO
  + w_distill * imitation_loss_coef * L_imit
  + encoder_loss_coef * L_enc
  + decoder_loss_coef * L_dec
```

- **PPO loss** — standard surrogate + value + entropy on the student's rollouts.
- **Imitation loss** — student mean action vs. teacher privileged action (behavior cloning).
- **Encoder reconstruction loss** (optional) — student vs. teacher encoder states.
- **Decoder reconstruction loss** (optional) — student vs. teacher decoder outputs; only active
  when the actor has a decoder.

**Key parameters:**

- `imitation_loss_coef`, `encoder_loss_coef`, `decoder_loss_coef` — per-term weights (encoder /
  decoder default to `0`, i.e. disabled).
- `loss_type` — `"mse"` or `"huber"`.
- `loss_schedule` — `"fixed"` (constant `w_ppo = w_distill = 1`) or `"curriculum"`, which
  anneals the imitation weight down and the PPO weight up over training (`total_iteration`
  controls the schedule horizon).
- Inherits PPO's RND, Symmetry, multi-GPU, and recurrent-policy support.

Uses a dedicated `"ppo_distillation"` rollout storage that holds both PPO transition data and
the teacher's privileged targets.



# Configuration Examples
To use custom-made NN architectures and algorithms, define config class and parse it to rsl_rl agent config files. 

## Model config
```python
# may defined in isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py

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

## Algorithm config

```python
# may defined in isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py

@configclass
class RslRlPpoAEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm with AutoEncoder."""

    class_name: str = "PPOAE"
    decoder_loss_coef: float = 1.0
    loss_type: Literal["mse", "huber"] = "mse"


@configclass
class RslRlPpoVAEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm with Variational AutoEncoder."""

    class_name: str = "PPOVAE"
    decoder_loss_coef: float = 1.0
    loss_type: Literal["mse", "huber"] = "mse"
    kl_loss_coef: float = 0.2
    kl_clip: float = 0.0


@configclass
class RslRlPpoDistillationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm with Variational AutoEncoder."""

    class_name: str = "PPODistillation"
    imitation_loss_coef: float = 1.0
    encoder_loss_coef: float = 1.0
    decoder_loss_coef: float = 1.0
    loss_type: Literal["mse", "huber"] = "mse"
    total_iteration: int = 0
    loss_schedule: Literal["fixed", "curriculum"] = "fixed"
```

## Full config in agent directory 

```python 
# PPO with VAE

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

    # VAE
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
    algorithm = RslRlPpoVAEAlgorithmCfg(
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
        decoder_loss_coef=1.0,
        loss_type="mse",
        kl_loss_coef=0.01,
        kl_clip=0.0,
    )

# Distillation PPO 

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
    # GRU student
    # student = RslRlRNNEncoderModelCfg(
    #     hidden_dims=[512, 256, 128],
    #     activation="elu",
    #     obs_normalization=False,
    #     distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    #     encoder_obs_set="proprioceptive_history",
    #     encoder_obs_normalization=False,
    #     rnn_type="gru",
    #     rnn_hidden_dim=64,
    #     rnn_num_layers=2,
    # )

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
        imitation_loss_coef=1.0,
        encoder_loss_coef=1.0,
        decoder_loss_coef=1.0,
        loss_type="mse",
        total_iteration=max_iterations,
        loss_schedule="curriculum",
        # loss_schedule="fixed",
    )

```