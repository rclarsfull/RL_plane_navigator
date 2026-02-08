"""
LunarLanderContinuous wrapper to test MaskedHybridPPO with a standard environment
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LunarLanderContinuousMultiOutput(gym.Wrapper):
    """
    Wraps LunarLanderContinuous to have a Dict action space matching the MaskedHybridPPO structure.
    
    Original LunarLanderContinuous actions:
    Box(-1, +1, (2,), dtype=np.float32)
    Action[0]: Main engine
        - < 0: Off
        - 0..1: Throttle 50%..100%
    Action[1]: Lateral boosters
        - -0.5..0.5: Off
        - < -0.5: Left booster (Throttle scales 50%..100%)
        - > 0.5: Right booster (Throttle scales 50%..100%)
    
    Multi-output structure (sorted alphabetically):
    - main_opt_switch: Discrete(2)
        0: Off
        1: On
    - main_val_power: Box(0, 1)
        Throttle value.
        clipped to 0..1. 
        Note: PPO outputs centered at 0. So init will be around 0 (Min Power).
    - side_opt_switch: Discrete(3)
        0: Off
        1: Left
        2: Right
    - side_val_power: Box(0, 1)
        Throttle value
        
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Create multi-output action space
        # Keys named to ensure alphabetical sorting aligns with logic:
        # 1. main_opt_switch
        # 2. main_val_power
        # 3. side_opt_switch
        # 4. side_val_power
        self.action_space = spaces.Dict({
            'main_opt_switch': spaces.Discrete(2),
            'main_val_power': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'side_opt_switch': spaces.Discrete(3),
            'side_val_power': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
    def step(self, action):
        """
        Map multi-output action to original LunarLanderContinuous action
        """
        if isinstance(action, dict):
            main_switch = action['main_opt_switch']
            main_power = action['main_val_power']
            side_switch = action['side_opt_switch']
            side_power = action['side_val_power']
        else:
            main_switch = int(action[0])
            main_power = float(action[1])
            side_switch = int(action[2])
            side_power = float(action[3])
            
        if isinstance(main_switch, np.ndarray):
            main_switch = main_switch.item()
        if isinstance(side_switch, np.ndarray):
            side_switch = side_switch.item()

        # Map Box(0, 1) handling from PPO Gaussian(0, std)
        # We simply clip. 
        # Mean 0 -> 0 -> Min throttle (50% real world)
        main_p = np.clip(main_power, 0.0, 1.0)
        side_p = np.clip(side_power, 0.0, 1.0)
        
        if isinstance(main_p, np.ndarray):
             main_p = main_p.item()
        if isinstance(side_p, np.ndarray):
             side_p = side_p.item()
            
        # 1. Handle Main Engine
        # Env logic: main < 0 is off. 0..1 is on.
        if main_switch == 0:
            actual_main = -1.0 # Safe off
        else:
            actual_main = main_p
                
        # 2. Handle Side Engines
        # Env logic: -0.5 < x < 0.5 off
        # x < -0.5 Left
        # x > 0.5 Right
        if side_switch == 0:
            actual_side = 0.0
        elif side_switch == 1: # Left
            # Map 0..1 to -0.5..-1.0
            actual_side = -0.5 - (side_p * 0.5)
        elif side_switch == 2: # Right
            # Map 0..1 to 0.5..1.0
            actual_side = 0.5 + (side_p * 0.5)
        else:
            actual_side = 0.0

        real_action = np.array([actual_main, actual_side], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = self.env.step(real_action)
        
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
