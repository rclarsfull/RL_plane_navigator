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
        if self.all_agents.is_active_agent_finished():
            raise Exception("Active agent has already finished, this should not happen")

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

    def _get_reward(self):
        agent = self.all_agents.get_active_agent()
        
        distance_to_waypoint = agent.distance_to_waypoint_normalized if agent.distance_to_waypoint_normalized != 0.0 else float(1e-6)
        drift_normalized = agent.drift / np.pi  # [0, 1]
        drift_reward = (1.0 - drift_normalized)**2 * DRIFT_FACTOR / (distance_to_waypoint/ 2.0)
        if drift_normalized > 0.5:
            drift_reward = -(drift_normalized)**2 * DRIFT_FACTOR / (distance_to_waypoint/ 2.0)
        

        collision_avoidance_reward = 0.0
        
        for slot_idx, intruder_id in enumerate(agent.selected_intruder_ids):
            if intruder_id not in agent.intruder_collision_cache:
                continue
            
            collision_info = agent.intruder_collision_cache[intruder_id]
            min_separation = collision_info['min_separation']
            closing_rate = collision_info['closing_rate']
            time_to_min_sep = collision_info['time_to_min_sep']
            

            if not self._is_actual_danger(closing_rate, time_to_min_sep, min_separation):
                continue
            
            if hasattr(agent, 'cpa_position_map') and intruder_id in agent.cpa_position_map:
                cpa_lat, cpa_lon = agent.cpa_position_map[intruder_id]
                
                if cpa_lat is not None and cpa_lon is not None:
                    ac_lat, ac_lon = agent.ac_lat, agent.ac_lon
                    _, distance_to_cpa = self.sim.geo_calculate_direction_and_distance(
                        ac_lat, ac_lon, cpa_lat, cpa_lon
                    )
                    
                    distance_from_cpa_normalized = np.clip(distance_to_cpa / OBS_DISTANCE, 0.0, 1.0)
                    time_urgency = np.clip(1.0 - (time_to_min_sep / LONG_CONFLICT_THRESHOLD_SEC), 0.0, 1.0)
                    
                    cpa_avoidance_reward = distance_from_cpa_normalized * time_urgency
                    collision_avoidance_reward -= cpa_avoidance_reward * CPA_WARNING_FACTOR
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[Step {self.steps}] CPA-BASED REWARD: {intruder_id} "
                                   f"| time_to_sep={time_to_min_sep:.1f}s | dist_to_cpa={distance_to_cpa:.2f}NM "
                                   f"| urgency={time_urgency:.2f} | reward={cpa_avoidance_reward:+.3f}")
    
        proximity_reward = 0.0

        waypoint = self.getNextWaypoint(agent)
        ac_lat, ac_lon = agent.ac_lat, agent.ac_lon
        _, distance_to_waypoint = self.sim.geo_calculate_direction_and_distance(ac_lat, ac_lon, waypoint.lat, waypoint.lon)
        
        distance_normalized = np.clip(distance_to_waypoint / OBS_DISTANCE, 0.0, 1.0)
        proximity_to_waypoint = 1.0 - distance_normalized  # [0, 1]: 1=ganz nah, 0=weit weg

        waypoint_factor = 1.0 + float(agent.waypoints_collected)

        proximity_reward = proximity_to_waypoint * PROXIMITY_REWARD_BASE * waypoint_factor
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Step {self.steps}] PROXIMITY REWARD: "
                        f"dist_to_wp={distance_to_waypoint:.2f}NM | proximity_norm={proximity_to_waypoint:.3f} | "
                        f"waypoints_collected={agent.waypoints_collected} | factor={waypoint_factor:.1f} | "
                        f"reward={proximity_reward:+.3f}")
        
            
        waypoint_bonus = agent.waypoint_reached_this_step * WAYPOINT_BONUS

        reward = (drift_reward + 
                 collision_avoidance_reward +
                 proximity_reward +
                 waypoint_bonus)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Step {self.steps}] REWARDS: "
                        f"Drift={drift_reward:+.3f} | "
                        f"CPA-Avoidance={collision_avoidance_reward:+.3f} | "
                        f"Proximity={proximity_reward:+.3f} | "
                        f"TOTAL={reward:+.3f}")
        

        agent.last_reward_components = {
            'drift': float(drift_reward),
            'cpa_warning': float(collision_avoidance_reward),
            'proximity': float(proximity_reward),
            'total': float(reward)
        }
        
        self.cumulative_drift_reward += drift_reward
        self.cumulative_cpa_warning_reward += collision_avoidance_reward
        self.cumulative_proximity_reward += proximity_reward
        self.total_reward += reward
        return reward

    def _set_action(self, action, agent: Agent) -> None:
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