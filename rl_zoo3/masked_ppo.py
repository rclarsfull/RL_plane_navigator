
import warnings
from typing import Any, Dict, Optional, Type, Union, Tuple

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
from stable_baselines3.common.distributions import CategoricalDistribution, DiagGaussianDistribution

STEER_INDEX = 2  # Index of the STEER action in the action_type discrete distribution

class MaskedMultiOutputDistribution(MultiOutputDistribution):
    """
    Custom Distribution that masks the log_prob of the steering head
    when the action_type is not steering (index 2).
    """
    #def log_prob(self, actions: th.Tensor) -> th.Tensor:
    #    # Split actions into components
    #    split_actions = th.split(actions, self.action_dims, dim=1)
#
    #    # Calculate log_prob for each component
    #    list_log_prob = [dist.log_prob(action) for dist, action in zip(self.distribution, split_actions)]
#
    #    # We assume the action space structure:
    #    # Index 0: action_type (Discrete)
    #    # Index 1: steer (Box)
    #    assert len(list_log_prob) == 2, "Expected exactly two action components"
#
    #    if len(list_log_prob) >= 2:
    #        # action_type is at index 0
    #        action_type_val = split_actions[0]
#
    #        # Create mask: 1.0 if action_type == 2 (STEER), 0.0 otherwise
    #        # action_type_val might be float, cast to long
    #        # squeeze() is needed because action_type_val has shape (batch_size, 1)
    #        is_steer = (action_type_val.long() == STEER_INDEX).float().squeeze(-1)
#
    #        # Apply mask to steer log_prob (index 1)
    #        # list_log_prob[1] has shape (batch_size,)
    #        list_log_prob[1] = list_log_prob[1] * is_steer
#
    #    log_prob = th.stack(list_log_prob, dim=1)
    #    log_prob = log_prob.sum(dim=1)
    #    return log_prob
#
    
    #def actions_from_params(
    #        self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False
    #) -> th.Tensor:
    #    """
    #    Returns samples from the probability distribution
    #    given its parameters.
#
    #    :param mean_actions:
    #    :param log_std:
    #    :param deterministic:
    #    :return: actions
    #    """
    #    split_mean_actions = th.split(mean_actions, self._net_action_dims, dim=1)
    #    split_log_std = th.split(log_std, self._net_action_dims, dim=1)
    #    list_actions = []
    #    assert len(self.distribution) == 2
    #    
    #    # Diskrete Action
    #    action_type_dist = self.distribution[0]
    #    assert isinstance(action_type_dist, CategoricalDistribution)
    #    action_type_mean = split_mean_actions[0]
    #    action_type = action_type_dist.actions_from_params(action_type_mean, deterministic=deterministic)
#
    #    # Kontinuierliche Action
    #    steer_dist = self.distribution[1]
    #    assert isinstance(steer_dist, DiagGaussianDistribution)
    #    steer_mean = split_mean_actions[1]
    #    steer_log_std = split_log_std[1]
#
    #    steer = steer_dist.actions_from_params(steer_mean, steer_log_std, deterministic=deterministic)
#
    #    # Gradient selektiv stoppen
    #    mask = (action_type == STEER_INDEX)  
    #    steer = th.where(mask, steer.detach(), steer)
#
    #    actions = th.cat([action_type, steer], dim=1)
    #    return actions
#
    #def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]: # hier noch machen
    #    """
    #    Returns samples and the associated log probabilities
    #    from the probability distribution given its parameters.
#
    #    :param mean_actions:
    #    :param log_std:
    #    :return: actions and log prob
    #    """
    #    split_mean_actions = th.split(mean_actions, self._net_action_dims, dim=1)
    #    split_log_std = th.split(log_std, self._net_action_dims, dim=1)
    #    list_actions = []
    #    list_log_prob = []
    #    for dist, dist_mean_actions, dist_log_std in zip(self.distribution, split_mean_actions, split_log_std):
    #        actions = None
    #        log_prob = None
    #        if isinstance(dist, (sb3.DiagGaussianDistribution, MultiOutputDistribution)):
    #            actions, log_prob = dist.log_prob_from_params(dist_mean_actions, dist_log_std)
    #        else:
    #            # For categorical and bernoulli distributions, mean actions are actually action logits
    #            actions, log_prob = dist.log_prob_from_params(dist_mean_actions)
    #        list_actions.append(actions)
    #        list_log_prob.append(log_prob)
#
    #    actions = th.cat(list_actions, dim=1)
    #    log_prob = th.cat(list_log_prob, dim=1)
    #    return actions, log_prob
    #
#    def log_prob(self, actions: th.Tensor) -> th.Tensor:
#        """
#        Returns the log probabilities of actions according to the distribution,
#        handling conditional actions explicitly like in the sample() method.
#
#        :param actions: the taken action
#        :return: The total log probability of the actions
#        """
#        assert len(self.distribution) == 2, "Expected exactly two action components"
#
#        action_type, steer = th.split(actions, self.action_dims, dim=1)
#
#        logp_action_type = self.distribution[0].log_prob(action_type)
#
#        is_steer_active = (action_type.long().squeeze(-1) == STEER_INDEX).unsqueeze(1)
#
#        # Log probability for steer: 0 for inactive ones
#        logp_steer = self.distribution[1].log_prob(steer)
#        logp_steer = logp_steer * is_steer_active.float()
#
#        total_log_prob = logp_action_type + logp_steer  #addition wegen mutiplikationsrgel log(a*b)=log(a)+log(b)
#
#        return total_log_prob
#    
#    def sample(self) -> th.Tensor:
#        assert len(self.distribution) == 2, "Expected exactly two action components"
#        # Sample action_type
#        action_type = self.distribution[0].sample()
#        if action_type.dim() == 1:
#            action_type = action_type.unsqueeze(1)
#
#        # Sample steer
#        steer = self.distribution[1].sample()
#        if steer.dim() == 1:
#            steer = steer.unsqueeze(1)
#
#        # Create mask: keep steer only when action_type == 2
#        is_steer = (action_type.long().squeeze(-1) == STEER_INDEX).float().unsqueeze(1)
#        steer = steer.where(is_steer == False, steer.detach())
#
#        return th.cat([action_type, steer], dim=1)
#
#    def entropy(self) -> Optional[th.Tensor]:
#        assert len(self.distribution) == 2, "Expected exactly two action components"
#        ent0 = self.distribution[0].entropy()
#        ent1 = self.distribution[1].entropy()
#        if ent0 is None or ent1 is None:
#            return None
#
#        # Versuche logits aus der torch.distributions.Categorical zu lesen
#        dist0 = self.distribution[0]
#        p_steer = None
#
#        # FlattenCategoricalDistribution hat ein torch.distributions.Categorical-Objekt
#        if hasattr(dist0, 'distribution') and hasattr(dist0.distribution, 'logits'):
#            logits = dist0.distribution.logits
#            p_steer = th.exp(F.log_softmax(logits, dim=-1))[..., STEER_INDEX]
#
#        assert p_steer is not None
#
#        if ent1.dim() > p_steer.dim():
#            p_steer = p_steer.unsqueeze(-1)
#
#        return ent0 + ent1 * p_steer
#





class MaskedMultiOutputActorCriticPolicy(MultiOutputActorCriticPolicy):
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                # TODO: what to do with Dict or Tuple space?
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
            
    #def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
    #    super().__init__(observation_space, action_space, lr_schedule, **kwargs)
#
    #    # Replace distribution with our custom masked distribution
    #    self.action_dist = MaskedMultiOutputDistribution(self.action_space)
#
    #    # Rebuild the action net to match the new distribution
    #    # (Though for MultiOutputDistribution it should be the same structure)
    #    latent_dim_pi = self.mlp_extractor.latent_dim_pi
    #    self.action_net, self.log_std = self.action_dist.proba_distribution_net(
    #        latent_dim=latent_dim_pi, log_std_init=self.log_std_init
    #    )
#
    #    # Re-initialize weights for the new action_net
    #    if self.ortho_init:
    #         module_gains = {self.action_net: 0.01}
    #         for module, gain in module_gains.items():
    #            module.apply(partial(self.init_weights, gain=gain))
#
    #    # Re-initialize the optimizer to include the new parameters
    #    self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
#
    #def forward_actor(self, obs: th.Tensor): Wie in: https://arxiv.org/pdf/2006.14171 Geht aber nicht bei mir weil ich nicht abhänig von dem env state maskiere sondern abhänig von der action
    #    latent_pi = self._get_latent_pi(obs)
        # Original logits
    #    logits = self.action_net(latent_pi)
    #    action_type_logits = logits[0]
    #    steer_logits = logits[1]
    #    action_type predicted logits → get argmax
    #    action_type = th.argmax(action_type_logits, dim=1) Das ist problematisch das dadurch zu früh angenommen wird das es die steer aktion ist
    #    Mask steer logits if action_type != 2
    #    mask = (action_type == 2).float().unsqueeze(1)
    #    steer_logits = steer_logits + (1 - mask) * -1e8
    #    return (action_type_logits, steer_logits)
    
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
