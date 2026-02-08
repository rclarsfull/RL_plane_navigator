import warnings
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union, Tuple

import torch as th
from gymnasium import spaces
import numpy as np

from sb3_plus import MultiOutputPPO
from sb3_plus.mimo.on_policy_algorithm import MultiOutputOnPolicyAlgorithm
from sb3_plus.mimo.policies import MultiOutputActorCriticPolicy, BaseFeaturesExtractor, FlattenExtractor
from sb3_plus.mimo.distributions import MultiOutputDistribution, FlattenCategoricalDistribution
from stable_baselines3.common import distributions as sb3_dist
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from sb3_plus.mimo.preprocessing import get_action_shape

MaskedSelfHybridPPO = TypeVar("MaskedSelfHybridPPO", bound="MaskedHybridPPO")

# Indices assuming sorted keys: 
# "main_opt_switch", "main_val_power", "side_opt_switch", "side_val_power"
IDX_MAIN_SWITCH = 0
IDX_MAIN_POWER = 1
IDX_SIDE_SWITCH = 2
IDX_SIDE_POWER = 3


class Masked_Hybrid_MultiOutputDistribution(MultiOutputDistribution):
    
    def make_proba_distribution(action_space: spaces.Space, use_sde: bool = False,
                            dist_kwargs: Optional[Dict[str, Any]] = None) -> sb3_dist.Distribution:
        if isinstance(action_space, (spaces.Dict, spaces.Tuple)):
            assert not use_sde, "Error: StateDependentNoiseDistribution not supported for multi action"
            return Masked_Hybrid_MultiOutputDistribution(action_space)
        elif isinstance(action_space, spaces.Discrete):
            return FlattenCategoricalDistribution(action_space)
        else:
            return sb3_dist.make_proba_distribution(action_space, use_sde, dist_kwargs)
    
    def stack_log_prob(self, actions: th.Tensor) -> th.Tensor:
        split_actions = th.split(actions, self.action_dims, dim=1)
        list_log_prob = [dist.log_prob(action) for dist, action in zip(self.distribution, split_actions)]
        log_prob = th.stack(list_log_prob, dim=1)
        return log_prob

    def stack_entropy(self) -> Optional[th.Tensor]:
        list_entropies = [dist.entropy() for dist in self.distribution]
        if None in list_entropies:
            return None
        return th.stack(list_entropies, dim=1)


class Masked_Hybrid_MultiOutputActorCriticPolicy(MultiOutputActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Dict,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[th.nn.Module] = th.nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.action_dist = Masked_Hybrid_MultiOutputDistribution.make_proba_distribution(action_space) 
        self._build(lr_schedule)
        
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        
        # Calculate masked log prob
        assert(isinstance(distribution, Masked_Hybrid_MultiOutputDistribution))
        
        # Split actions to get the discrete parts for masking
        # action_dims usually: [1, 1, 1, 1] for [Discrete(2), Box(1), Discrete(3), Box(1)]
        # Discrete actions come as floats in the tensor
        split_actions = th.split(actions, distribution.action_dims, dim=1)
        
        act_main_switch = split_actions[IDX_MAIN_SWITCH].squeeze(-1) # (batch,)
        act_side_switch = split_actions[IDX_SIDE_SWITCH].squeeze(-1) # (batch,)
        
        # Get all log probs
        stack_log_prob = distribution.stack_log_prob(actions)
        
        lp_main_switch = stack_log_prob[:, IDX_MAIN_SWITCH]
        lp_main_power = stack_log_prob[:, IDX_MAIN_POWER]
        lp_side_switch = stack_log_prob[:, IDX_SIDE_SWITCH]
        lp_side_power = stack_log_prob[:, IDX_SIDE_POWER]
        
        # Define masks
        # Main Switch: 0=Off, 1=On. If Off, power logic doesn't matter (masked out)
        mask_main = (act_main_switch.round().long() == 1)
        
        # Side Switch: 0=Off, 1=Left, 2=Right. If Off, power logic doesn't matter
        mask_side = (act_side_switch.round().long() != 0)
        
        # Apply masks
        # If masked (condition not met), we replace log_prob with 0
        masked_lp_main_power = th.where(mask_main, lp_main_power, th.zeros_like(lp_main_power))
        masked_lp_side_power = th.where(mask_side, lp_side_power, th.zeros_like(lp_side_power))
        
        # Sum up valid log probs
        masked_log_prob = lp_main_switch + masked_lp_main_power + lp_side_switch + masked_lp_side_power
        
        # Reshape actions for return ??? usually (batch, sum_dims) or shaped...
        # SB3 standard is usually flattened for Dict spaces in PPO unless specialized
        # user's masked_ppo.py calls reshape:
        actions = actions.reshape((-1, *get_action_shape(self.action_space)))
        
        return actions, values, masked_log_prob
    
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)
        assert(isinstance(distribution, Masked_Hybrid_MultiOutputDistribution))
        
        # Split actions
        split_actions = th.split(actions, distribution.action_dims, dim=1)
        
        act_main_switch = split_actions[IDX_MAIN_SWITCH].squeeze(-1)
        act_side_switch = split_actions[IDX_SIDE_SWITCH].squeeze(-1)
        
        # Log Probs
        stack_log_prob = distribution.stack_log_prob(actions)
        lp_main_switch = stack_log_prob[:, IDX_MAIN_SWITCH]
        lp_main_power = stack_log_prob[:, IDX_MAIN_POWER]
        lp_side_switch = stack_log_prob[:, IDX_SIDE_SWITCH]
        lp_side_power = stack_log_prob[:, IDX_SIDE_POWER]
        
        # Masks
        mask_main = (act_main_switch.round().long() == 1)
        mask_side = (act_side_switch.round().long() != 0)
        
        # Masked Log Probs
        masked_lp_main_power = th.where(mask_main, lp_main_power, th.zeros_like(lp_main_power))
        masked_lp_side_power = th.where(mask_side, lp_side_power, th.zeros_like(lp_side_power))
        
        masked_log_prob = lp_main_switch + masked_lp_main_power + lp_side_switch + masked_lp_side_power
    
        
        stack_entropy = distribution.stack_entropy()
        ent_main_switch = stack_entropy[:, IDX_MAIN_SWITCH]
        ent_main_power = stack_entropy[:, IDX_MAIN_POWER]
        ent_side_switch = stack_entropy[:, IDX_SIDE_SWITCH]
        ent_side_power = stack_entropy[:, IDX_SIDE_POWER]
        
        with th.no_grad():
            main_switch_probs = distribution.distribution[IDX_MAIN_SWITCH].distribution.probs
            side_switch_probs = distribution.distribution[IDX_SIDE_SWITCH].distribution.probs
            p_main_switch = main_switch_probs[:, 1]
            p_side_switch = 1 - side_switch_probs[:, 0]
        
        masked_ent_main_power =  p_main_switch * ent_main_power 
        masked_ent_side_power =  p_side_switch * ent_side_power 
        
        masked_entropy = ent_main_switch + masked_ent_main_power + ent_side_switch + masked_ent_side_power
    
        return values, masked_log_prob, masked_entropy


class MaskedHybridPPO(MultiOutputPPO):
    
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MaskedHybridPolicy": Masked_Hybrid_MultiOutputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy], Type[Masked_Hybrid_MultiOutputActorCriticPolicy]],
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
        MultiOutputOnPolicyAlgorithm.__init__(self,
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

        if (policy not in ["MaskedHybridPolicy",  Masked_Hybrid_MultiOutputActorCriticPolicy]
            and isinstance(self.action_space, (spaces.Dict, spaces.Tuple))
        ):
            raise ValueError(f"You must use `MaskedHybridPolicy` or `Masked_Hybrid_MultiOutputActorCriticPolicy` when working with "
                             f"dict or tuple action space, not {policy}")

        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1"

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()
            
    
    def learn(
            self: MaskedSelfHybridPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "Masked_Hybrid_PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> MaskedSelfHybridPPO:
        MultiOutputOnPolicyAlgorithm.learn(
            self=self,
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        return self
