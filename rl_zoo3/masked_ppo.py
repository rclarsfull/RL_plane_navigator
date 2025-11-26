
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from functools import partial

from sb3_plus import MultiOutputPPO
from sb3_plus.mimo.policies import MultiOutputActorCriticPolicy, MIMOActorCriticPolicy
from sb3_plus.mimo.distributions import MultiOutputDistribution
from sb3_plus.mimo.on_policy_algorithm import MultiOutputOnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_schedule_fn, explained_variance
#class MaskedMultiOutputDistribution(MultiOutputDistribution):
#    """
#    Custom Distribution that masks the log_prob of the steering head
#    when the action_type is not steering (index 2).
#    """
#    def log_prob(self, actions: th.Tensor) -> th.Tensor:
#        # Split actions into components
#        split_actions = th.split(actions, self.action_dims, dim=1)
#        
#        # Calculate log_prob for each component
#        list_log_prob = [dist.log_prob(action) for dist, action in zip(self.distribution, split_actions)]
#        
#        # We assume the action space structure:
#        # Index 0: action_type (Discrete)
#        # Index 1: steer (Box)
#        assert len(list_log_prob) == 2, "Expected exactly two action components"
#        
#        if len(list_log_prob) >= 2:
#            # action_type is at index 0
#            action_type_val = split_actions[0]
#            
#            # Create mask: 1.0 if action_type == 2 (STEER), 0.0 otherwise
#            # action_type_val might be float, cast to long
#            # squeeze() is needed because action_type_val has shape (batch_size, 1)
#            is_steer = (action_type_val.long() == 2).float().squeeze(-1)
#            
#            # Apply mask to steer log_prob (index 1)
#            # list_log_prob[1] has shape (batch_size,)
#            list_log_prob[1] = list_log_prob[1] * is_steer
#            
#        log_prob = th.stack(list_log_prob, dim=1)
#        log_prob = log_prob.sum(dim=1)
#        return log_prob

#class MaskedMultiOutputActorCriticPolicy(MultiOutputActorCriticPolicy):
#    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
#        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
#        
#        # Replace distribution with our custom masked distribution
#        self.action_dist = MaskedMultiOutputDistribution(self.action_space)
#        
#        # Rebuild the action net to match the new distribution
#        # (Though for MultiOutputDistribution it should be the same structure)
#        latent_dim_pi = self.mlp_extractor.latent_dim_pi
#        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
#            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
#        )
#        
#        # Re-initialize weights for the new action_net
#        if self.ortho_init:
#             module_gains = {self.action_net: 0.01}
#             for module, gain in module_gains.items():
#                module.apply(partial(self.init_weights, gain=gain))
#        
#        # Re-initialize the optimizer to include the new parameters
#        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

class MaskedMultiOutputActorCriticPolicy(MultiOutputActorCriticPolicy):
    def forward_actor(self, obs: th.Tensor):
        latent_pi = self._get_latent_pi(obs)

        # Original logits
        logits = self.action_net(latent_pi)

        action_type_logits = logits[0]
        steer_logits = logits[1]

        # action_type predicted logits → get argmax
        action_type = th.argmax(action_type_logits, dim=1)

        # Mask steer logits if action_type != 2
        mask = (action_type == 2).float().unsqueeze(1)
        steer_logits = steer_logits + (1 - mask) * -1e8

        return (action_type_logits, steer_logits)
    
class MaskedMultiOutputPPO(MultiOutputPPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy], Type[MultiOutputActorCriticPolicy]],
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
        # Use our custom policy if generic one is requested
        if policy in ["MultiOutputPolicy", "MIMOPolicy", MultiOutputActorCriticPolicy, MIMOActorCriticPolicy]:
            policy = MaskedMultiOutputActorCriticPolicy
            
        # Call grandparent __init__ to bypass the policy check in MultiOutputPPO
        MultiOutputOnPolicyAlgorithm.__init__(
            self,
            policy=policy,
            env=env,
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
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
                spaces.Dict,
                spaces.Tuple,
            ),
        )

        # Copy-paste from MultiOutputPPO.__init__ (skipping the check)
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
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
