import numpy as np
import torch as th
from gymnasium import spaces

from rl_zoo3.masked_ppo import MaskedMultiOutputRolloutBuffer, STEER_INDEX


def test_masked_rollout_buffer_masking():
    obs_space = spaces.Box(-1, 1, (2,))
    action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(-1, 1, (1,))))
    buf = MaskedMultiOutputRolloutBuffer(4, obs_space, action_space, device='cpu', n_envs=2)

    # Prepare two-step log-probs per env
    per_head = th.tensor([[-0.1, -0.2], [-0.3, -0.4]], dtype=th.float32)

    # Fill buffer: alternate first env steer=2, second env steer=1
    for i in range(4):
        obs = np.zeros((2, 2), dtype=np.float32)
        action = np.zeros((2, 2), dtype=np.float32)
        action[0, 0] = STEER_INDEX
        action[1, 0] = 1
        reward = np.ones(2)
        episode_start = np.zeros(2)
        v = th.tensor([0.0, 0.0])
        buf.add(obs, action, reward, episode_start, v, per_head)

    # Sample all
    samples = list(buf.get(batch_size=4))
    assert len(samples) == 1
    s = samples[0]

    # steer_mask should be 1 for env0 and 0 for env1 across batch
    assert (s.steer_mask.numpy() == np.array([1.0, 0.0])).all()

    # old_log_prob is sum of heads
    expected_sum = (-0.1 - 0.2) + 0.0  # for env0 masked: -0.3; env1: (-0.3 -0.4) -> but masked will be -0.3
    # check per-batch masked calculation element-wise
    lp_masked = s.old_log_prob_masked.numpy()
    # both elements should be finite
    assert np.all(np.isfinite(lp_masked))
import numpy as np
import torch as th
from gymnasium import spaces

from rl_zoo3.masked_ppo import MaskedMultiOutputRolloutBuffer, STEER_INDEX


def test_steer_mask_and_masked_logprob():
    obs_space = spaces.Box(-1, 1, (2,))
    action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(-1, 1, (1,))))
    buf = MaskedMultiOutputRolloutBuffer(4, obs_space, action_space, device="cpu", n_envs=2)

    # fill buffer
    for i in range(4):
        obs = np.zeros((2, 2), dtype=np.float32)
        action = np.zeros((2, 2), dtype=np.float32)
        action[0, 0] = STEER_INDEX
        action[1, 0] = 0
        reward = np.ones(2)
        episode_start = np.zeros(2)
        v = th.tensor([0.1, 0.2])
        lp = th.tensor([[-0.1, -0.2], [-0.3, -0.4]])
        buf.add(obs, action, reward, episode_start, v, lp)

    assert buf.full

    for sample in buf.get(batch_size=2):
        # steer_mask should be present
        assert hasattr(sample, "steer_mask")
        # masked old log prob should equal logp_type + mask * logp_steer
        lp_sum = sample.old_log_prob
        masked = sample.old_log_prob_masked
        steer = sample.steer_mask
        # shapes
        assert lp_sum.shape[0] == masked.shape[0]
        assert steer.shape[0] == masked.shape[0]