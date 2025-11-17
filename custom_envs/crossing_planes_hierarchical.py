"""
HIERARCHICAL 2-LEVEL SYSTEM

HIGH-LEVEL AGENT:
- 3 Discrete Actions:
  0: Steuern (query low-level model for continuous action)
  1: NoOp (continue with last action)
  2: Direct Route (set heading directly to target)

LOW-LEVEL MODEL:
- Trained SAC model from crossing_planes_conti.py
- Provides continuous heading/speed adjustments

ARCHITECTURE:
High-Level Agent (THIS ENV) -> Decision every AGENT_INTERACTION_TIME
  |
  +-> Action 0: Query Low-Level Model (SAC) -> Apply continuous action
  +-> Action 1: NoOp -> Continue last action
  +-> Action 2: Direct Route -> Set heading to waypoint
"""

import gymnasium as gym
import numpy as np
import pygame
import random
import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path

from .helper_classes import Waypoint, Agent, Camera, Agents
from .consts import *
from .helper_functions import (
    compute_cpa_multi_heading_numba,
    bound_angle_positive_negative_180,
    get_point_at_distance
)
from .base_crossing_env import BaseCrossingEnv, HEADING_OFFSETS, NUM_HEADING_OFFSETS
from simulator.blue_sky_adapter import Simulator

# Import für Low-Level Model Loading
from stable_baselines3 import SAC

logger = logging.getLogger(__name__)

CPA_WARNING_FACTOR = 15.0
WAYPOINT_BONUS = 80.0

class CrossingPlanesHierarchical(gym.Env, BaseCrossingEnv):
    """
    HIERARCHICAL ENVIRONMENT: High-Level Decision Making
    
    Action Space: Discrete(3)
    - 0: Steuern (use low-level model)
    - 1: NoOp (last action continues)
    - 2: Direct Route (set heading to target)
    
    Observation Space: Same as crossing_planes_conti.py
    """
    
    metadata = {
        "name": "crossing_planes_hierarchical",
        "render_modes": ["rgb_array", "human"],
        "render_fps": 120
    }
    
    def __init__(self, render_mode=None, low_level_model_path=None):
        """
        Args:
            render_mode: 'human' or 'rgb_array'
            low_level_model_path: Path to trained SAC model (.zip file)
        """
        if render_mode is not None:
            assert render_mode in self.metadata["render_modes"], f"Invalid render_mode {render_mode}"
        
        self.step_limit = int(TIME_LIMIT / AGENT_INTERACTION_TIME)
        self.resets_counter = 0
        
        # Initialize BaseCrossingEnv (creates simulator, camera, and rendering setup internally)
        BaseCrossingEnv.__init__(self, CENTER_LAT, CENTER_LON, render_mode=render_mode)
        
        # Environment state
        self.steps = 0
        
        # Reward tracking
        self.cumulative_drift_reward = 0.0
        self.cumulative_intrusion_reward = 0.0
        self.cumulative_cpa_warning_reward = 0.0
        self.cumulative_waypoint_bonus = 0.0
        self.cumulative_proximity_reward = 0.0
        self.cumulative_speed_stability_reward = 0.0
        self.cumulative_action_selection_bonus = 0.0
        
        # OBSERVATION SPACE (same as continuous version)
        obs_dim = 6 + NUM_AC_STATE * 8 + NUM_HEADING_OFFSETS
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # ACTION SPACE: Discrete(3)
        # 0 = Steuern (use low-level model)
        # 1 = NoOp (continue last action)
        # 2 = Direct Route (set heading to target)
        self.action_space = gym.spaces.Discrete(3)
        
        # Episode statistics
        self.total_reward = 0
        self.num_episodes = 0
        self.total_intrusions = 0
        
        # === LOW-LEVEL MODEL LOADING ===
        self.low_level_model = None
        self.low_level_model_path = low_level_model_path
        
        if low_level_model_path is not None:
            self._load_low_level_model(low_level_model_path)
        
        # Track last continuous action for NoOp
        self.last_continuous_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # Action statistics for logging
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        logger.info(f"[HIERARCHICAL ENV] Initialized with action space: Discrete(3)")
        logger.info(f"  Action 0: Steuern (Low-Level Model)")
        logger.info(f"  Action 1: NoOp (Last Action)")
        logger.info(f"  Action 2: Direct Route")
    
    def _load_low_level_model(self, model_path: str):
        """Load trained SAC model for action 0 (Steuern)"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"[LOW-LEVEL MODEL] File not found: {model_path}")
                return
            
            logger.info(f"[LOW-LEVEL MODEL] Loading from: {model_path}")
            self.low_level_model = SAC.load(str(model_path))
            logger.info(f"[LOW-LEVEL MODEL] Successfully loaded!")
            
        except Exception as e:
            logger.error(f"[LOW-LEVEL MODEL] Failed to load: {e}")
            self.low_level_model = None
    
    def _query_low_level_model(self, observation: np.ndarray) -> np.ndarray:
        """
        Query low-level SAC model for continuous action.
        
        Args:
            observation: Current observation (same format as continuous env)
        
        Returns:
            continuous_action: [heading_change, speed_change] in [-1, 1]
        """
        if self.low_level_model is None:
            logger.warning("[LOW-LEVEL MODEL] Not loaded! Using random action.")
            return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        
        try:
            # SAC model prediction (deterministic=False for exploration)
            action, _ = self.low_level_model.predict(observation, deterministic=False)
            return action.astype(np.float32)
        
        except Exception as e:
            logger.error(f"[LOW-LEVEL MODEL] Prediction failed: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)
    
    # ==================== COPY METHODS FROM CONTINUOUS ENV ====================
    # (All observation/reward/collision methods identical to crossing_planes_conti.py)
    

    def _get_reward(self):
        """
        HIERARCHICAL REWARD:
        Same base rewards as continuous env, plus bonus for efficient action selection
        """
        agent = self.all_agents.get_active_agent()
        
        # Base rewards (copied from continuous env)
        distance_to_waypoint = agent.distance_to_waypoint_normalized if agent.distance_to_waypoint_normalized != 0.0 else float(1e-6)
        drift_normalized = agent.drift / np.pi
        drift_reward = (1.0 - drift_normalized)**2 * DRIFT_FACTOR / (distance_to_waypoint / 2.0)
        if drift_normalized > 0.5:
            drift_reward = -(drift_normalized)**2 * DRIFT_FACTOR / (distance_to_waypoint / 2.0)
        
        collision_avoidance_reward = 0.0
        
        proximity_reward = 0.0
        waypoint = self.getNextWaypoint(agent)
        ac_lat, ac_lon = agent.ac_lat, agent.ac_lon
        _, distance_to_waypoint = self.sim.geo_calculate_direction_and_distance(ac_lat, ac_lon, waypoint.lat, waypoint.lon)
        distance_normalized = np.clip(distance_to_waypoint / OBS_DISTANCE, 0.0, 1.0)
        proximity_to_waypoint = 1.0 - distance_normalized
        waypoint_factor = 1.0 + float(agent.waypoints_collected)
        proximity_reward = proximity_to_waypoint * PROXIMITY_REWARD_BASE * waypoint_factor
        
        speed_stability_reward = 0.0
        waypoint_bonus = agent.waypoint_reached_this_step * WAYPOINT_BONUS
        
        # Action selection bonus (encourage efficient use of low-level model)
        action_selection_bonus = 0.0
        if hasattr(agent, 'last_high_level_action'):
            # Bonus for using NoOp when drift is small (efficient)
            if agent.last_high_level_action == 1 and drift_normalized < 0.3:
                action_selection_bonus = 0.05
            # Bonus for using Direct Route when no danger and close to waypoint
            elif agent.last_high_level_action == 2 and distance_normalized < 0.2:
                action_selection_bonus = 0.03
        
        reward = (drift_reward + 
                 collision_avoidance_reward +
                 proximity_reward +
                 speed_stability_reward +
                 waypoint_bonus +
                 action_selection_bonus)
        
        agent.last_reward_components = {
            'drift': float(drift_reward),
            'cpa_warning': float(collision_avoidance_reward),
            'proximity': float(proximity_reward),
            'speed_stability': float(speed_stability_reward),
            'action_selection': float(action_selection_bonus),
            'total': float(reward)
        }
        
        self.cumulative_drift_reward += drift_reward
        self.cumulative_cpa_warning_reward += collision_avoidance_reward
        self.cumulative_proximity_reward += proximity_reward
        self.cumulative_speed_stability_reward += speed_stability_reward
        self.cumulative_action_selection_bonus += action_selection_bonus
        self.total_reward += reward
        return reward
    
    def _get_info(self):
        """Get episode info dict"""
        active_agent = self.all_agents.get_active_agent()
        last_reward_components = active_agent.last_reward_components if active_agent.last_reward_components else {}
        
        return {
            'avg_reward': float(self.total_reward / self.steps if self.steps > 0 else 0),
            'intrusions': self.total_intrusions,
            'drift_active_agent': float(active_agent.drift * 180 / np.pi),
            'agent_finished': bool(self.all_agents.is_active_agent_finished()),
            'is_success': bool(self.all_agents.is_active_agent_finished() and self.total_intrusions == 0),
            'steps': int(self.steps),
            'waypoints_collected': int(active_agent.waypoints_collected),
            'num_episodes': int(self.num_episodes),
            'action_counts': dict(self.action_counts),
            'last_high_level_action': int(active_agent.last_high_level_action) if hasattr(active_agent, 'last_high_level_action') else -1,
            **last_reward_components
        }
    
    # ==================== HIERARCHICAL ACTION EXECUTION ====================
    
    def _set_action(self, high_level_action: int, agent: Agent) -> None:
        """
        Execute high-level action:
        - 0: Steuern (query low-level model)
        - 1: NoOp (continue last action)
        - 2: Direct Route (set heading to target)
        """
        self.action_counts[high_level_action] += 1
        agent.last_high_level_action = high_level_action
        
        if high_level_action == 0:
            # ACTION 0: STEUERN - Query Low-Level Model
            observation = self._get_observation(agent)
            continuous_action = self._query_low_level_model(observation)
            self.last_continuous_action = continuous_action
            self._apply_continuous_action(continuous_action, agent)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Step {self.steps}] Action 0 (STEUERN): Low-Level Model -> {continuous_action}")
        
        elif high_level_action == 1:
            # ACTION 1: NOOP - Do Nothing, just let simulator continue
            # No interaction with sim, aircraft continues with current heading/speed
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[Step {self.steps}] Action 1 (NOOP): No action, simulator continues")
        
        elif high_level_action == 2:
            # ACTION 2: DIRECT ROUTE - Set Heading to Waypoint
            waypoint = self.getNextWaypoint(agent)
            ac_lat, ac_lon = agent.ac_lat, agent.ac_lon
            target_heading, _ = self.sim.geo_calculate_direction_and_distance(
                ac_lat, ac_lon, waypoint.lat, waypoint.lon
            )
            
            _, _, current_heading, _ = self.sim.traf_get_state(agent.id)
            self.sim.traf_set_heading(agent.id, target_heading)
            agent.last_set_heading = target_heading
            
            # Keep speed constant (no speed change)
            self.last_continuous_action = np.array([0.0, 0.0], dtype=np.float32)
            
            if logger.isEnabledFor(logging.DEBUG):
                import custom_envs.helper_functions as fn
                logger.debug(f"[Step {self.steps}] Action 2 (DIRECT ROUTE): {current_heading:.1f}° -> {target_heading:.1f}°")
    
    def _apply_continuous_action(self, action: np.ndarray, agent: Agent) -> None:
        """
        Apply continuous action [heading_change, speed_change] to agent.
        (Same as continuous env)
        """
        heading_change = float(np.clip(action[0], -1.0, 1.0))
        speed_change = float(np.clip(action[1], -1.0, 1.0))
        
        # Heading Change
        dh = heading_change * D_HEADING
        _, _, current_heading, _ = self.sim.traf_get_state(agent.id)
        heading_new = bound_angle_positive_negative_180(current_heading + dh)
        self.sim.traf_set_heading(agent.id, heading_new)
        agent.last_set_heading = heading_new
        
        # Speed Change
        _, _, _, current_speed = self.sim.traf_get_state(agent.id)
        speed_delta = speed_change * D_SPEED
        new_speed = np.clip(current_speed + speed_delta, MIN_SPEED, MAX_SPEED)
        self.sim.traf_set_speed(agent.id, new_speed)
    
    # ==================== EPISODE MANAGEMENT ====================
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        self.resets_counter += 1
        
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
        
        # Reset reward tracking
        self.cumulative_drift_reward = 0.0
        self.cumulative_intrusion_reward = 0.0
        self.cumulative_cpa_warning_reward = 0.0
        self.cumulative_waypoint_bonus = 0.0
        self.cumulative_proximity_reward = 0.0
        self.cumulative_speed_stability_reward = 0.0
        self.cumulative_action_selection_bonus = 0.0
        
        # Reset action statistics
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.last_continuous_action = np.array([0.0, 0.0], dtype=np.float32)
        
        logger.debug(f"Generating aircraft for episode #{self.num_episodes}...")
        self._gen_aircraft(self.num_episodes)
        if self.render_mode is not None:
            self.agent_trails = {agent.id: [] for agent in self.all_agents.get_all_agents()}
        
        logger.debug(f"Computing initial observation...")
        observations = self._get_observation()
        infos = self._get_info()
        
        logger.debug(f"Episode #{self.num_episodes} initialized. Observation shape: {observations.shape}")
        return observations, infos
    
    def step(self, action: int):
        """
        Execute one step with high-level action.
        
        Args:
            action: High-level discrete action (0, 1, or 2)
        
        Returns:
            observation, reward, done, truncate, info
        """
        if self.all_agents.is_active_agent_finished():
            raise Exception("Active agent has already finished")
        
        aktiv_agent = self.all_agents.get_active_agent()
        self._set_action(action, aktiv_agent)
        
        self.sim.sim_step(float(AGENT_INTERACTION_TIME))
        
        for agent in self.all_agents.get_all_agents():
            agent.check_finish(self)
            agent.update_observation_cache(self)
        
        aktiv_agent.check_intrusion(self)
        
        done = self.all_agents.is_active_agent_finished()
        observation = self._get_observation()
        reward = self._get_reward()
        info = self._get_info()
        truncate = (self.steps >= self.step_limit) or self.total_intrusions > 0
        
        self.steps += 1
        return observation, reward, done, truncate, info
    
    # ==================== AIRCRAFT GENERATION (same as continuous env) ====================
