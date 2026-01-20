"""
LunarLander wrapper to test MaskedMultiOutputPPO with a standard environment
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LunarLanderMultiOutput(gym.Wrapper):
    """
    Wraps LunarLander to have a Dict action space matching your masked PPO structure:
    
    Original LunarLander actions:
    0: do nothing
    1: fire left orientation engine
    2: fire main engine
    3: fire right orientation engine
    
    Multi-output structure:
    - action_type: Discrete(3)
        0: NOOP (do nothing)
        1: MAIN_ENGINE (fire main engine)
        2: SIDE_ENGINE (fire left or right orientation engine) <- STEER_INDEX
    - side_direction: Discrete(2)
        0: left orientation engine
        1: right orientation engine
        (only relevant when action_type == 2)
    
    This allows testing your MaskedMultiOutputPPO on a standard environment
    where conditional actions make sense.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Create multi-output action space
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(3),     # 0=NOOP, 1=MAIN_ENGINE, 2=SIDE_ENGINE
            'side_direction': spaces.Discrete(2)   # 0=left, 1=right
        })
        
    def step(self, action):
        """
        Map multi-output action to original LunarLander action:
        - action_type=0 -> 0 (NOOP)
        - action_type=1 -> 2 (MAIN_ENGINE)
        - action_type=2 + side_direction=0 -> 1 (LEFT)
        - action_type=2 + side_direction=1 -> 3 (RIGHT)
        """
        # Action comes as numpy array [action_type, side_direction]
        if isinstance(action, dict):
            action_type = action['action_type']
            side_direction = action['side_direction']
        else:
            # Assuming action is a numpy array or list
            action_type = int(action[0])
            side_direction = int(action[1])
        
        if action_type == 0:
            # NOOP
            actual_action = 0
        elif action_type == 1:
            # MAIN_ENGINE
            actual_action = 2
        elif action_type == 2:
            # SIDE_ENGINE - here the side_direction matters (STEER_INDEX)
            if side_direction == 0:
                actual_action = 1  # left
            else:
                actual_action = 3  # right
        else:
            actual_action = 0
            
        obs, reward, terminated, truncated, info = self.env.step(actual_action)
        
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
