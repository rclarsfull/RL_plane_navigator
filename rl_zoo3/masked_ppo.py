
import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Tuple


import torch as th
from gymnasium import spaces


from sb3_plus import MultiOutputPPO
from sb3_plus.mimo.policies import MultiOutputActorCriticPolicy, MIMOActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.policies import (
    ActorCriticPolicy, BasePolicy
)


STEER_INDEX = 2  # Index of the STEER action in the action_type discrete distribution

class Masked_MultiOutputActorCriticPolicy(MultiOutputActorCriticPolicy):
    
    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # ----- Value -----
        values = self.value_net(latent_vf)

        # ===== High level: action type =====
        logits_type = self.type_head(latent_pi)
        dist_type = th.distributions.Categorical(logits=logits_type)

        if deterministic:
            action_type = th.argmax(logits_type, dim=1)
        else:
            action_type = dist_type.sample()

        logp_type = dist_type.log_prob(action_type)

        # ===== Low level: steer (conditional) =====
        batch_size = obs.shape[0]
        steer = th.zeros(batch_size, 1, device=obs.device)
        logp_steer = th.zeros(batch_size, device=obs.device)

        steer_mask = action_type == STEER_INDEX

        if steer_mask.any():
            latent_s = latent_pi[steer_mask]
            mean, log_std = self.steer_head(latent_s)
            std = th.exp(log_std)

            dist_steer = th.distributions.Normal(mean, std)

            if deterministic:
                steer_val = mean
            else:
                steer_val = dist_steer.rsample()

            steer[steer_mask] = steer_val
            logp_steer[steer_mask] = dist_steer.log_prob(steer_val).sum(dim=1)

        # ===== Joint action & log-prob =====
        actions = th.cat([action_type.unsqueeze(1), steer], dim=1)
        logp_steer_masked = logp_steer[steer_mask]  # nur die relevanten
        log_prob = logp_type.clone()
        log_prob[steer_mask] += logp_steer_masked

        return actions, values, log_prob

    
    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # ----- Value -----
        values = self.value_net(latent_vf).flatten()

        # Split actions
        action_type = actions[:, 0].long()
        steer_action = actions[:, 1:2]

        # ===== High-level log-prob =====
        logits_type = self.type_head(latent_pi)
        dist_type = th.distributions.Categorical(logits=logits_type)

        logp_type = dist_type.log_prob(action_type)
        entropy_type = dist_type.entropy()

        # ===== Low-level (conditional) =====
        batch_size = obs.shape[0]
        logp_steer = th.zeros(batch_size, device=obs.device)
        entropy_steer = th.zeros(batch_size, device=obs.device)

        steer_mask = action_type == STEER_INDEX

        if steer_mask.any():
            latent_s = latent_pi[steer_mask]
            mean, log_std = self.steer_head(latent_s)
            std = th.exp(log_std)

            dist_steer = th.distributions.Normal(mean, std)

            steer_val = steer_action[steer_mask]
            logp_steer[steer_mask] = dist_steer.log_prob(steer_val).sum(dim=1)
            entropy_steer[steer_mask] = dist_steer.entropy().sum(dim=1)

        # ===== Joint outputs =====
        logp_steer_masked = logp_steer[steer_mask]  # nur die relevanten
        log_prob = logp_type.clone()
        log_prob[steer_mask] += logp_steer_masked

        entropy_steer_masked = entropy_steer[steer_mask]  # nur die relevanten
        entropy = entropy_type.clone()
        entropy[steer_mask] += entropy_steer_masked

        return values, log_prob, entropy


     

class MaskedMultiOutputPPO(MultiOutputPPO):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiOutputPolicy": Masked_MultiOutputActorCriticPolicy
    }
    
    supported_action_spaces: ClassVar[Tuple] = (
        spaces.Box,
        spaces.Discrete,
        spaces.MultiDiscrete,
        spaces.MultiBinary,
        spaces.Dict,
        spaces.Tuple,
    )

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy], Type[Masked_MultiOutputActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
        )

        if (policy not in ["MultiOutputPolicy", "MIMOPolicy", MultiOutputActorCriticPolicy, MIMOActorCriticPolicy]
            and isinstance(self.action_space, (spaces.Dict, spaces.Tuple))
        ):
            raise ValueError(f"You must use `MultiOutputPolicy` or `MIMOPolicy` when working with "
                             f"dict or tuple action space, not {policy}")

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()