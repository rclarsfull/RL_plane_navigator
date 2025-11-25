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
from enum import IntEnum

import gymnasium as gym
import numpy as np
import pygame

import bluesky_gym.envs.common.functions as fn
from typing import Dict, List, Optional
from collections import deque

from .helper_classes import Waypoint, Agent, Agents
from .consts import *
from .helper_functions import bound_angle_positive_negative_180, get_point_at_distance
from .base_crossing_env import BaseCrossingEnv

logger = logging.getLogger(__name__)


class CrossingPlanesMultiHead(gym.Env, BaseCrossingEnv):
    metadata = {
        "name": "merge_v1",
        "render_modes": ["rgb_array","human"],
         "render_fps": 120}

    def __init__(self, render_mode=None):
        # Initialize base class
        BaseCrossingEnv.__init__(
            self, 
            center_lat=CENTER_LAT, 
            center_lon=CENTER_LON, 
            render_mode=render_mode,
            window_width=1500,
            window_height=1000
        )

        if render_mode is not None:
            assert render_mode in self.metadata["render_modes"], f"Invalid render_mode {render_mode}"
        
        self.step_limit = int(TIME_LIMIT / AGENT_INTERACTION_TIME)
        self.resets_counter = 0

        self.steps = 0
        self.actions_noop_count = 0
        self.actions_steer_count = 0
        self.last_continuous_action = 0.0
        
        self.cumulative_drift_reward = 0.0
        self.cumulative_intrusion_reward = 0.0
        self.cumulative_action_age_reward = 0.0
        self.cumulative_cpa_warning_reward = 0.0
        self.cumulative_waypoint_bonus = 0.0
        self.cumulative_proximity_reward = 0.0
        self.cumulative_noop_reward = 0.0

        self.action_markers = {}

        # +1: zusätzlicher CPA-Feature-Eintrag für Ziel-Heading
        obs_dim = 10 + NUM_AC_STATE * 9 + (NUM_HEADING_OFFSETS + 1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action Space: Dict with discrete action type and continuous steering
        # - action_type: 0=NOOP, 1=SNAP, 2=STEER
        # - steer: continuous in [-1, 1] mapped to [-180°, +180°] when action_type==2
        self.action_space = gym.spaces.Dict({
            'action_type': gym.spaces.Discrete(3),
            'steer': gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        self.total_reward = 0.0
        self.num_episodes = 0
        self.total_intrusions = 0

        # Define Enum for action types for clearer intent
        class ActionType(IntEnum):
            NOOP = 0
            SNAP = 1
            STEER = 2
        self.ActionType = ActionType

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
        self.actions_noop_count = 0
        self.actions_steer_count = 0
        self.last_continuous_action = 0.0

        self.cumulative_drift_reward = 0.0
        self.cumulative_intrusion_reward = 0.0
        self.cumulative_action_age_reward = 0.0
        self.cumulative_cpa_warning_reward = 0.0
        self.cumulative_waypoint_bonus = 0.0
        self.cumulative_proximity_reward = 0.0
        self.cumulative_noop_reward = 0.0
        
        logger.debug(f"Generating aircraft for episode #{self.num_episodes}...")
        self._gen_aircraft()
        self.agent_trails = {agent.id: [] for agent in self.all_agents.get_all_agents()}
        self.action_markers = {agent.id: [] for agent in self.all_agents.get_all_agents()}
        for agent in self.all_agents.get_all_agents():
            agent.action_markers_with_steering = []

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
    
    def _gen_aircraft(self):
        """Wrapper method that calls base class implementation"""
        BaseCrossingEnv._gen_aircraft(self, self.num_episodes)

    def _get_observation(self, agent: Optional[Agent] = None) -> np.ndarray:
        """Wrapper - uses base class implementation"""
        return BaseCrossingEnv._get_observation(self, agent)

    def _get_info(self):
        active_agent = self.all_agents.get_active_agent()

        total_action_age = sum(agent.action_age for agent in self.all_agents.agents)
        avg_action_age = total_action_age / len(self.all_agents.agents) if len(self.all_agents.agents) > 0 else 0
        avg_action_age_seconds = avg_action_age * AGENT_INTERACTION_TIME
        
        last_reward_components = active_agent.last_reward_components if active_agent.last_reward_components else {}
        
        return {
            'avg_reward': float(self.total_reward/self.steps if self.steps > 0 else 0),
            'intrusions': self.total_intrusions,
            'drift_active_agent': float(active_agent.drift * 180 / np.pi),
            'agent_finished': bool(self.all_agents.is_active_agent_finished()),
            'is_success': bool(self.all_agents.is_active_agent_finished() and self.total_intrusions == 0),
            'steps': int(self.steps),
            'waypoints_collected': int(active_agent.waypoints_collected),
            'actions_noop': int(self.actions_noop_count),
            'actions_steer': int(self.actions_steer_count),
            'last_continuous_action': float(self.last_continuous_action),
            'num_episodes': int(self.num_episodes),
            'cumulative_drift_reward': float(self.cumulative_drift_reward),
            'cumulative_action_age_reward': float(self.cumulative_action_age_reward),
            'cumulative_cpa_warning_reward': float(self.cumulative_cpa_warning_reward),
            'cumulative_waypoint_bonus': float(self.cumulative_waypoint_bonus),
            'cumulative_proximity_reward': float(self.cumulative_proximity_reward),
            'cumulative_noop_reward': float(self.cumulative_noop_reward),
            'avg_action_age': float(avg_action_age),
            'avg_action_age_seconds': float(avg_action_age_seconds),
            'total_cumulative_reward': float(self.cumulative_drift_reward + self.cumulative_intrusion_reward + 
                                              self.cumulative_action_age_reward + self.cumulative_cpa_warning_reward +
                                              self.cumulative_waypoint_bonus +
                                              self.cumulative_proximity_reward +
                                              self.cumulative_noop_reward),

            'last_reward_drift': float(last_reward_components.get('drift', 0.0)),
            'last_reward_action_age': float(last_reward_components.get('action_age', 0.0)),
            'last_reward_cpa_warning': float(last_reward_components.get('cpa_warning', 0.0)),
            'last_reward_proximity': float(last_reward_components.get('proximity', 0.0)),
            'last_reward_noop': float(last_reward_components.get('noop', 0.0)),
            'last_reward_total': float(last_reward_components.get('total', 0.0))
        }

    def _get_reward(self):
        """
        DENSE REWARD STRUCTURE - 6 KOMPONENTEN!
        
        Alle Komponenten geben Feedback basierend auf Agent-Verhalten:
        
        DENSE COMPONENTS (jeden Step):
        1. drift_reward: [0, 0.6] (abhängig vom Heading Error zu Ziel) ← JEDEN STEP!
        2. action_age_reward: [0, 0.1] (abhängig von NOOP-Dauer) ← JEDEN STEP (wenn NOOP)!
        3. collision_avoidance_reward: [0, 0.5] (abhängig von Abstand zu CPA) ← NUR bei Gefahr!
        4. proximity_reward: [0, ?] (abhängig von Nähe zum Waypoint + Anzahl gesammelter Waypoints) ← JEDEN STEP!
        5. noop_reward: [NOOP_REWARD] (flache Belohnung für jede NOOP-Aktion) ← JEDEN STEP (wenn NOOP)!
        
        """
        agent = self.all_agents.get_active_agent()
        
        distance_to_waypoint = agent.distance_to_waypoint_normalized if agent.distance_to_waypoint_normalized != 0.0 else float(1e-6)
        drift_normalized = agent.drift / np.pi  # [0, 1]
        drift_reward = (1.0 - drift_normalized)**2 * DRIFT_FACTOR / (distance_to_waypoint/ 2.0)
        
        action_age_reward = 0.0
        if agent.is_noop:
            action_age_seconds = agent.action_age * AGENT_INTERACTION_TIME
            action_age_minutes = action_age_seconds / 60.0
            action_age_normalized = np.clip(action_age_minutes / 15.0, 0.0, 1.0)
            action_age_reward = action_age_normalized * ACTION_AGE_FACTOR
        
        collision_avoidance_reward = 0.0
        obs_dict = agent.obs_dict_cache if agent.obs_dict_cache is not None else self._get_observation_dict(agent)
        
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
        
        noop_reward = 0.0
        if agent.is_noop:
            noop_reward = NOOP_REWARD
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Step {self.steps}] NOOP REWARD: "
                        f"is_noop={agent.is_noop} | "
                        f"reward={noop_reward:+.3f}")
            
        waypoint_bonus = agent.waypoint_reached_this_step * WAYPOINT_BONUS

        reward = (drift_reward + 
                 action_age_reward +
                 collision_avoidance_reward +
                 proximity_reward +
                 noop_reward +
                 waypoint_bonus)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Step {self.steps}] REWARDS: "
                        f"Drift={drift_reward:+.3f} | "
                        f"ActionAge={action_age_reward:+.3f} | "
                        f"CPA-Avoidance={collision_avoidance_reward:+.3f} | "
                        f"Proximity={proximity_reward:+.3f} | "
                        f"NOOP={noop_reward:+.3f} | "
                        f"TOTAL={reward:+.3f}")
        

        agent.last_reward_components = {
            'drift': float(drift_reward),
            'action_age': float(action_age_reward),
            'cpa_warning': float(collision_avoidance_reward),
            'proximity': float(proximity_reward),
            'noop': float(noop_reward),
            'total': float(reward)
        }
        
        self.cumulative_drift_reward += drift_reward
        self.cumulative_action_age_reward += action_age_reward
        self.cumulative_cpa_warning_reward += collision_avoidance_reward
        self.cumulative_proximity_reward += proximity_reward
        self.cumulative_noop_reward += noop_reward
        self.total_reward += reward
        return reward


    def _set_action(self, action, agent: Agent) -> None:
        """Parse and apply action.

        Supported formats:
        - Dict: {'action_type': int, 'steer': float/array}
        - Sequence/np.ndarray: [action_type, steer]

        action_type (Enum):
            ActionType.NOOP  (0)  -> no change
            ActionType.SNAP  (1)  -> snap heading to next waypoint bearing
            ActionType.STEER (2)  -> apply relative heading change (steer * 180°)

        steer: continuous in [-1,1]. Only used when STEER.
        """
        BaseCrossingEnv._set_action(self, action, agent)
        ActionType = self.ActionType

        # ---- Extract raw components ----
        if isinstance(action, dict):
            raw_type = action['action_type']
            raw_steer = action['steer']
        elif isinstance(action, (list, tuple, np.ndarray)):
            if len(action) < 2:
                raise ValueError(f"Sequence action length must be >=2, got {len(action)}")
            raw_type = action[0]
            raw_steer = action[1]
        else:
            raise TypeError(f"Unsupported action container {type(action)}")

        try:
            action_type = ActionType(int(raw_type))
        except ValueError:
            raise ValueError(f"Invalid action_type {raw_type}, expected 0,1,2")

        if isinstance(raw_steer, (list, tuple, np.ndarray)):
            steer_scalar = float(np.asarray(raw_steer).reshape(-1)[0])
        else:
            steer_scalar = float(raw_steer)
        steer_scalar = float(np.clip(steer_scalar, -1.0, 1.0))

        is_active_agent = (agent == self.all_agents.get_active_agent())

        # Für Render Logging: Hole aktuelle Position (falls nötig)
        if self.render_mode is not None:
            lat, lon, _, _ = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(self.sim, lat, lon, self.window_width, self.window_height)
        else:
            x_pos, y_pos = None, None

        if action_type == ActionType.NOOP:  # NOOP
            # NOOP Aktion
            agent.is_noop = True
            agent.last_action = 0
            agent.last_action_continuous = 0.0
            agent.last_action_type = 'noop'
            self.last_continuous_action = 0.0
            if is_active_agent:
                self.actions_noop_count += 1
            if self.render_mode is not None and x_pos is not None:
                if agent.id in self.action_markers:
                    self.action_markers[agent.id].append((x_pos, y_pos, 'noop'))
                    if len(self.action_markers[agent.id]) > MAX_ACTION_MARKERS:
                        self.action_markers[agent.id].pop(0)
                if hasattr(agent, 'action_markers_with_steering'):
                    agent.action_markers_with_steering.append((x_pos, y_pos, 'noop', 0.0, agent.action_age))
                    if len(agent.action_markers_with_steering) > MAX_ACTION_MARKERS:
                        agent.action_markers_with_steering.pop(0)
            return

        # SNAP oder STEER Aktion
        agent.is_noop = False
        agent.last_action = 1
        if is_active_agent:
            if action_type == ActionType.SNAP:
                # optional separater Counter für Snap
                if not hasattr(self, 'actions_snap_count'):
                    self.actions_snap_count = 0
                self.actions_snap_count += 1
            elif action_type == ActionType.STEER:
                self.actions_steer_count += 1

        _, _, current_heading, _ = self.sim.traf_get_state(agent.id)

        if action_type == ActionType.SNAP:  # SNAP
            waypoint = self.getNextWaypoint(agent)
            target_dir, _ = self.sim.geo_calculate_direction_and_distance(agent.ac_lat, agent.ac_lon, waypoint.lat, waypoint.lon)
            heading_new = bound_angle_positive_negative_180(target_dir)
            rel_delta = bound_angle_positive_negative_180(heading_new - current_heading)
            continuous_steering = float(np.clip(rel_delta / 180.0, -1.0, 1.0))
            agent.last_action_type = 'snap'
        elif action_type == ActionType.STEER:  # STEER (continuous)
            rel_deg = steer_scalar * 180.0
            heading_new = bound_angle_positive_negative_180(current_heading + rel_deg)
            continuous_steering = float(np.clip(rel_deg / 180.0, -1.0, 1.0))
            agent.last_action_type = 'steer'
        else:
            # Fallback (sollte nicht auftreten, da oben NOOP abgefangen)
            heading_new = current_heading
            continuous_steering = 0.0
            agent.last_action_type = 'noop'

        # Setze Heading im Simulator
        self.sim.traf_set_heading(agent.id, heading_new)
        agent.last_set_heading = heading_new
        agent.action_age = 0
        agent.last_action_continuous = continuous_steering
        self.last_continuous_action = continuous_steering

        if self.render_mode is not None and x_pos is not None:
            marker_type = agent.last_action_type
            if agent.id in self.action_markers:
                self.action_markers[agent.id].append((x_pos, y_pos, marker_type))
                if len(self.action_markers[agent.id]) > MAX_ACTION_MARKERS:
                    self.action_markers[agent.id].pop(0)
            if hasattr(agent, 'action_markers_with_steering'):
                agent.action_markers_with_steering.append((x_pos, y_pos, marker_type, continuous_steering, agent.action_age))
                if len(agent.action_markers_with_steering) > MAX_ACTION_MARKERS:
                    agent.action_markers_with_steering.pop(0)
            
    def render(self):
        BaseCrossingEnv.render(self,self.render_mode)
        
    def close(self):
        BaseCrossingEnv.close(self)

    
