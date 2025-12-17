
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union, Tuple


import torch as th
from gymnasium import spaces


from sb3_plus import MultiOutputPPO
from sb3_plus.mimo.policies import MultiOutputActorCriticPolicy, MIMOActorCriticPolicy, BaseFeaturesExtractor, FlattenExtractor
from sb3_plus.mimo.distributions import MultiOutputDistribution, FlattenCategoricalDistribution
from stable_baselines3.common import distributions as sb3_dist
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.policies import (
    ActorCriticPolicy, BasePolicy
)
from sb3_plus.mimo.preprocessing import get_action_shape

STEER_INDEX = 2  # Index of the STEER action in the action_type discrete distribution
MaskedSelfMultiOutputPPO = TypeVar("MaskedSelfMultiOutputPPO", bound="MaskedMultiOutputPPO")

class Masked_MultiOutputDistribution(MultiOutputDistribution):
    
    def make_proba_distribution(action_space: spaces.Space, use_sde: bool = False,
                            dist_kwargs: Optional[Dict[str, Any]] = None) -> sb3_dist.Distribution:
        """
        Return an instance of Distribution for the correct type of action space
        :param action_space: the input action space
        :param use_sde: Force the use of StateDependentNoiseDistribution
            instead of DiagGaussianDistribution
        :param dist_kwargs: Keyword arguments to pass to the probability distribution
        :return: the appropriate Distribution object
        """
        if isinstance(action_space, (spaces.Dict, spaces.Tuple)):
            assert not use_sde, "Error: StateDependentNoiseDistribution not supported for multi action"
            return Masked_MultiOutputDistribution(action_space)
        elif isinstance(action_space, spaces.Discrete):
            return FlattenCategoricalDistribution(action_space)
        else:
            return sb3_dist.make_proba_distribution(action_space, use_sde, dist_kwargs)
    
    def stack_log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Returns the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions: the taken action
        :return: The log likelihood of the distribution
        """
        split_actions = th.split(actions, self.action_dims, dim=1)
        list_log_prob = [dist.log_prob(action) for dist, action in zip(self.distribution, split_actions)]
        log_prob = th.stack(list_log_prob, dim=1)
        return log_prob

    def stack_entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability
        :return: the entropy, or None if no analytical form is known
        """
        list_entropies = [dist.entropy() for dist in self.distribution]
        if None in list_entropies:
            return None
        return th.stack(list_entropies, dim=1)

class Masked_MultiOutputActorCriticPolicy(MultiOutputActorCriticPolicy):
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
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            share_features_extractor=share_features_extractor
            
        )

        # Action distribution
        self.action_dist = Masked_MultiOutputDistribution.make_proba_distribution(action_space) 

        self._build(lr_schedule)
   
    

class MaskedMultiOutputPPO(MultiOutputPPO):
    
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiOutputPolicy": MultiOutputActorCriticPolicy,
    }

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

        if (policy not in ["MultiOutputPolicy",  Masked_MultiOutputActorCriticPolicy]
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
            
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        assert(isinstance(distribution, Masked_MultiOutputDistribution), "This should be a Masked Distribution")
        type_action = actions[:, 0]
        steer_action = actions[:, 1]
        mask = (type_action == STEER_INDEX).float()
        
        stack_log_prob = distribution.stack_log_prob(actions)
        action_type_log_prob = stack_log_prob[:,0]
        steer_log_prob = stack_log_prob[:,1]
        masked_log_prob = action_type_log_prob + mask * steer_log_prob
        
        actions = actions.reshape((-1, *get_action_shape(self.action_space)))
        return actions, values, masked_log_prob
    
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            
        distribution = self._get_action_dist_from_latent(latent_pi)
        assert(isinstance(distribution, Masked_MultiOutputDistribution), "This should be a Masked Distribution")
        type_action = actions[:, 0]
        steer_action = actions[:, 1]
        mask = (type_action == STEER_INDEX).float()
        
        stack_log_prob = distribution.stack_log_prob(actions)
        action_type_log_prob = stack_log_prob[:,0]
        steer_log_prob = stack_log_prob[:,1]
        masked_log_prob = action_type_log_prob + mask * steer_log_prob
        
        values = self.value_net(latent_vf)
        
        stack_entropy = distribution.stack_entropy()
        action_type_entropy = stack_entropy[:,0]
        steer_entropy = stack_entropy[:,1]
        masked_entropy = action_type_entropy + mask * steer_entropy
        
        return values, masked_log_prob, masked_entropy
    
    def learn(
            self: MaskedSelfMultiOutputPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "Masked_MO_PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> MaskedSelfMultiOutputPPO:
        super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        return self

    