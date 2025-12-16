
import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union, Tuple, NamedTuple, Generator

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from functools import partial

from sb3_plus import MultiOutputPPO
from sb3_plus.mimo.buffers import MultiOutputRolloutBuffer
from sb3_plus.mimo.policies import MultiOutputActorCriticPolicy, MIMOActorCriticPolicy
from sb3_plus.mimo.distributions import MultiOutputDistribution
from sb3_plus.mimo.on_policy_algorithm import MultiOutputOnPolicyAlgorithm
from sb3_plus.mimo.buffers import MultiOutputDictRolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from stable_baselines3.common.distributions import CategoricalDistribution, DiagGaussianDistribution
from stable_baselines3.common.utils import obs_as_tensor
from sb3_plus.mimo.preprocessing import clip_actions


STEER_INDEX = 2  # Index of the STEER action in the action_type discrete distribution

SelfMaskedMultiOutputPPO = TypeVar("SelfMaskedMultiOutputPPO", bound="MaskedMultiOutputPPO")

class MaskedMultiOutputActorCriticPolicy(MultiOutputActorCriticPolicy):
    """Policy subclass kept intact; training loop lives on the algorithm side.

    Note: Do NOT override the Module.train() signature here, as
    `set_training_mode` expects to call the standard `train(mode: bool)`.
    """
    pass
            
class MaskedMultiOutputRolloutBuffer(MultiOutputRolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    steer_mask: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        # Call parent reset first to ensure base fields like pos/full are reset
        super().reset()
        # Then override/create our specialized buffers (so parent reset does not overwrite them)
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # log_probs: store per-head log-probs for multi-output actions
        if isinstance(self.action_space, (spaces.Dict, spaces.Tuple)):
            n_heads = len(self.action_space.spaces) if isinstance(self.action_space, spaces.Tuple) else len(self.action_space.spaces)
        else:
            raise AssertionError("MaskedMultiOutputRolloutBuffer requires a multi-output action space (Dict or Tuple)")
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, n_heads), dtype=np.float32)
        # Mask whether steering action was selected (1.0) or not (0.0)
        self.steer_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        # store steer mask: action[:, 0] == STEER_INDEX -> float mask
        a = np.array(action)
        if a.ndim == 1:
            a = a.reshape((self.n_envs, -1))
        if a.shape[1] < 1:
            raise AssertionError("Action shape incompatible for MaskedMultiOutputRolloutBuffer; expected at least one head")
        # expect the first head to indicate action_type (discrete)
        self.steer_mask[self.pos] = (a[:, 0] == STEER_INDEX).astype(np.float32)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator["MaskedRolloutBufferSamples", None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "steer_mask",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    class MaskedRolloutBufferSamples(NamedTuple):
        observations: th.Tensor
        actions: th.Tensor
        old_values: th.Tensor
        old_log_prob: th.Tensor
        advantages: th.Tensor
        returns: th.Tensor
        steer_mask: th.Tensor
        old_log_prob_masked: th.Tensor

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> "MaskedRolloutBufferSamples":
        # log_probs stored per-head (e.g. [logp_type, logp_steer])
        lp = self.log_probs[batch_inds]
        # If stored as flat scalar (backwards compatibility), make it 2D
        if lp.ndim == 1:
            lp = lp.reshape(-1, 1)

        # Sum of per-head old log-probs (compatible with previous API)
        old_log_prob_sum = lp.sum(axis=1)

        # steer_mask for this batch
        steer = self.steer_mask[batch_inds].reshape(-1)

        # Compute masked old_log_prob: logp_type + mask * logp_steer
        if lp.shape[1] < 2:
            raise AssertionError("Expected at least 2 action heads (type, steer) for MaskedMultiOutputRolloutBuffer")
        old_log_prob_masked = lp[:, 0] + steer * lp[:, 1]

        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            old_log_prob_sum,
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            steer,
            old_log_prob_masked,
        )

        torched = tuple(map(self.to_torch, data))
        return MaskedMultiOutputRolloutBuffer.MaskedRolloutBufferSamples(*torched)
    

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

    def _setup_model(self) -> None:
        """Set up model and use MaskedMultiOutputRolloutBuffer instead of the default one."""
        # setup lr schedule and seed like parent
        self._setup_lr_schedule()
        # Ensure clip range and clip_range_vf are wrapped as FloatSchedule
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)
        self.set_random_seed(self.seed)

        buffer_cls = MaskedMultiOutputRolloutBuffer # Buffer class replacement

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Instantiate policy as usual
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
    
    def train(self) -> None:
        """Perform one PPO update using the masked rollout buffer.

        This method implements the training loop that was previously (incorrectly)
        placed on the policy. It computes per-head log-probs and entropies, builds
        masked log-probs/entropy for the steering head and applies the PPO loss.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Defensive: ensure clip_range and clip_range_vf are wrapped as FloatSchedule
        if not callable(self.clip_range):
            self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None and not callable(self.clip_range_vf):
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)
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
                # Get values from policy
                values, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Compute per-head log-probs and entropy (don't detach steering head)
                distribution = self.policy.get_distribution(rollout_data.observations)
                # Ensure actions shape is (batch, *) when splitting
                if actions.dim() == 1:
                    actions = actions.unsqueeze(1)
                split_actions = th.split(actions, distribution.action_dims, dim=1)
                list_lp = [dist.log_prob(a) for dist, a in zip(distribution.distribution, split_actions)]
                logp_per_head = th.stack(list_lp, dim=1)
                logp_type = logp_per_head[:, 0]
                logp_steer = logp_per_head[:, 1] 

                # per-head entropy
                list_ent = [dist.entropy() for dist in distribution.distribution]
                if None in list_ent:
                    entropy_per_head = None
                else:
                    entropy_per_head = th.stack(list_ent, dim=1)
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # mask for steering from rollout (tensor on correct device)
                mask = rollout_data.steer_mask

                # masked log-prob: type + mask * steer
                # use detach for steering log-prob when mask is 0 to prevent gradient flow
                
                logp_steer_no_grad = logp_steer.detach()
                logp_steer_masked = th.where(mask.bool(), logp_steer, logp_steer_no_grad)
                logp = logp_type + logp_steer_masked

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(logp - rollout_data.old_log_prob_masked)

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

                # Entropy loss favor exploration, masked analog
                if entropy_per_head is None:
                    # We require per-head entropy to compute masked entropy correctly
                    raise AssertionError(
                        "Per-head entropy not available: masking requires analytical entropy for each action head"
                    )
                else:
                    entropy_type = entropy_per_head[:, 0]
                    entropy_steer = entropy_per_head[:, 1] if entropy_per_head.shape[1] >= 2 else th.zeros_like(entropy_type)
                    entropy_loss = -th.mean(entropy_type + mask * entropy_steer)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = logp - rollout_data.old_log_prob_masked
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

            
    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: MaskedMultiOutputRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                # Compute per-head log-probs so the buffer stores raw components
                try:
                    distribution = self.policy.get_distribution(obs_tensor)
                    split_actions = th.split(actions, distribution.action_dims, dim=1)
                    list_lp = [dist.log_prob(a) for dist, a in zip(distribution.distribution, split_actions)]
                    per_head_logprob = th.stack(list_lp, dim=1)
                except Exception:
                    # We require per-head log-probs (one per action head) for masking
                    raise AssertionError("Failed to compute per-head log-probs: policy must expose per-head distributions")
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, (spaces.Box, spaces.Dict)):
                clipped_actions = clip_actions(actions, self.action_space)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            # TODO: what to do for Dict space ?
            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values, per_head_logprob,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def learn(
            self: SelfMaskedMultiOutputPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "Masked_MO_PPO", #Set new default tensorboard log name
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfMaskedMultiOutputPPO:
        super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        return self


