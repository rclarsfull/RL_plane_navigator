"""
BS Komandos: https://github.com/TUDelft-CNS-ATM/bluesky/wiki/Command-Reference

Bis 3.11
- Mit dynamischert dt geht nicht, weil widerspricht gegen allgemeien andanhme im RL siehe discount factor
- Latzte action in den Obs space, eventuell auch rotationsgeschwindichkeit
- Belohnungen anpassen starke aktionen nicht stärker bestrafen.
- Gedanken mach zu richtige NO OP
-- Im obs space zielheading der letzten op, und aktionen updaten nicht direkt den sim sondern dieses ziel heading.
- Vill berrechenn wievie von der letzten aktion bereits erledigt,

"""

import logging
import random
import gymnasium as gym
import numpy as np
import bluesky_gym.envs.common.functions as fn
from .helper_classes import Agent
from .consts import *
from .helper_functions import (
    bound_angle_positive_negative_180,
    get_point_at_distance,
    compute_cpa_multi_heading_numba
)
from .base_crossing_env import BaseCrossingEnv, HEADING_OFFSETS, NUM_HEADING_OFFSETS
from simulator.blue_sky_adapter import Simulator

CPA_WARNING_FACTOR = 15.0
WAYPOINT_BONUS = 80.0

class Cross_env(BaseCrossingEnv, gym.Env):
    metadata = {
        "name": "merge_v1",
        "render_modes": ["rgb_array","human"],
         "render_fps": 120}

    def __init__(self, render_mode=None):

        if render_mode is not None:
            assert render_mode in self.metadata["render_modes"], f"Invalid render_mode {render_mode}"
        
        self.step_limit = int(TIME_LIMIT / AGENT_INTERACTION_TIME)
        self.resets_counter = 0

        # Initialize BaseCrossingEnv (creates simulator, camera, and rendering setup internally)
        super().__init__(CENTER_LAT, CENTER_LON, render_mode=render_mode)

        self.steps = 0
        
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.total_reward = 0
        self.num_episodes = 0
        self.total_intrusions = 0  
    
    def step(self, action):
        """Wrapper - uses base class implementation"""
        aktiv_agent = self.all_agents.get_active_agent()
        self._set_action(action, aktiv_agent)
        return BaseCrossingEnv._step(self, action)
    
    def reset(self, seed=None, options=None):
        return BaseCrossingEnv.reset(self, seed=seed, options=options)
    

    def _compute_action_history_refactored(self, agent: Agent) -> np.ndarray:
        """
        REMOVED: Action history no longer needed in continuous action space.
        Returns empty array for compatibility.
        """
        return np.array([], dtype=np.float32)



    def _get_reward(self) -> float:
        """Wrapper - uses base class implementation"""
        return BaseCrossingEnv._get_reward(self)

    def _set_action(self, action, agent: Agent) -> None:
        BaseCrossingEnv._set_action(self, action, agent)
        heading_change = float(np.clip(action[0], -1.0, 1.0))

        dh = heading_change * D_HEADING
        _, _, current_heading, _ = self.sim.traf_get_state(agent.id)
        heading_new = fn.bound_angle_positive_negative_180(current_heading + dh)
        self.sim.traf_set_heading(agent.id, heading_new)
        agent.last_set_heading = heading_new
    
    def render(self):
        BaseCrossingEnv.render(self, self.render_mode)
        
    def close(self):
        BaseCrossingEnv.close(self)