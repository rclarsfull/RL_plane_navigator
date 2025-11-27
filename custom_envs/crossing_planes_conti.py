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
import time

import gymnasium as gym
import numpy as np
import pygame

import bluesky_gym.envs.common.functions as fn
from typing import Dict, List, Optional
from collections import deque

from .helper_classes import Waypoint, Agent, Camera, Agents
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

class Cross_env(gym.Env, BaseCrossingEnv):
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
        BaseCrossingEnv.__init__(self, CENTER_LAT, CENTER_LON, render_mode=render_mode)

        self.steps = 0
        
        self.cumulative_drift_reward = 0.0
        self.cumulative_intrusion_reward = 0.0
        self.cumulative_cpa_warning_reward = 0.0
        self.cumulative_waypoint_bonus = 0.0
        self.cumulative_proximity_reward = 0.0

        # Observation Space Breakdown:
        # ego_state: 6 (heading_cos, heading_sin, speed, drift, v_sep, dist_to_wp)
        # threat_features: NUM_AC_STATE * 8 = 4 * 8 = 32
        # multi_heading_cpa: NUM_HEADING_OFFSETS (min time_to_cpa for each heading offset) = 9
        # Total: 6 + 32 + 9 = 47
        obs_dim = 5 + NUM_AC_STATE * 11 + NUM_HEADING_OFFSETS
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.total_reward = 0
        self.num_episodes = 0
        self.total_intrusions = 0  

    def reset(self, seed=None, options=None):
        self.resets_counter += 1
        
        # ⚡ Logging nur bei Debug-Level (reduziert Overhead beim Training)
        if self.all_agents is not None and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n{'='*100}")
            logger.debug(f"RESET #{self.resets_counter} - Previous Episode Summary:")
            logger.debug(self._get_info())
            logger.debug(f"{'='*100}\n")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.sim.traf_reset()
        self.steps = 0
 
        self.num_episodes += 1
        self.total_reward = 0
        self.total_intrusions = 0

        self.cumulative_drift_reward = 0.0
        self.cumulative_intrusion_reward = 0.0
        self.cumulative_cpa_warning_reward = 0.0
        self.cumulative_waypoint_bonus = 0.0
        self.cumulative_proximity_reward = 0.0
        
        logger.debug(f"Generating aircraft for episode #{self.num_episodes}...")
        self._gen_aircraft(self.num_episodes)
        if self.render_mode is not None:
            self.agent_trails = {agent.id: [] for agent in self.all_agents.get_all_agents()}
        
        logger.debug(f"Computing initial observation...")
        observations = self._get_observation()
        infos = self._get_info()
        
        logger.debug(f"Episode #{self.num_episodes} initialized. Initial observation shape: {observations.shape}")

        return observations, infos
    
    def step(self, action):
        """Wrapper - uses base class implementation"""
        return BaseCrossingEnv._step(self, action)
    

    def _compute_action_history_refactored(self, agent: Agent) -> np.ndarray:
        """
        REMOVED: Action history no longer needed in continuous action space.
        Returns empty array for compatibility.
        """
        return np.array([], dtype=np.float32)

    def _get_info(self):
        active_agent = self.all_agents.get_active_agent()
        
        last_reward_components = active_agent.last_reward_components if active_agent.last_reward_components else {}
        
        return {
            'avg_reward': float(self.total_reward/self.steps if self.steps > 0 else 0),
            'intrusions': self.total_intrusions,
            'drift_active_agent': float(active_agent.drift * 180 / np.pi),
            'agent_finished': bool(self.all_agents.is_active_agent_finished()),
            'is_success': bool(self.all_agents.is_active_agent_finished() and self.total_intrusions == 0),
            'steps': int(self.steps),
            'waypoints_collected': int(active_agent.waypoints_collected),
            'num_episodes': int(self.num_episodes),
            'cumulative_drift_reward': float(self.cumulative_drift_reward),
            'cumulative_cpa_warning_reward': float(self.cumulative_cpa_warning_reward),
            'cumulative_waypoint_bonus': float(self.cumulative_waypoint_bonus),
            'cumulative_proximity_reward': float(self.cumulative_proximity_reward),
            'total_cumulative_reward': float(self.cumulative_drift_reward + self.cumulative_intrusion_reward + 
                                              self.cumulative_cpa_warning_reward +
                                              self.cumulative_waypoint_bonus +
                                              self.cumulative_proximity_reward),

            'last_reward_drift': float(last_reward_components.get('drift', 0.0)),
            'last_reward_cpa_warning': float(last_reward_components.get('cpa_warning', 0.0)),
            'last_reward_proximity': float(last_reward_components.get('proximity', 0.0)),
            'last_reward_total': float(last_reward_components.get('total', 0.0))
        }

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
        BaseCrossingEnv.render(self,self.render_mode)
        
    def close(self):
        BaseCrossingEnv.close(self)