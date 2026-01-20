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


class CrossingPlanesMultiHead(BaseCrossingEnv, gym.Env):
    metadata = {
        "name": "merge_v1",
        "render_modes": ["rgb_array","human"],
         "render_fps": 120}

    def __init__(self, render_mode=None):
        # Initialize base class
        super().__init__(
            center_lat=CENTER_LAT, 
            center_lon=CENTER_LON, 
            render_mode=render_mode,
            window_width=1500,
            window_height=1000
        )

        if render_mode is not None:
            assert render_mode in self.metadata["render_modes"], f"Invalid render_mode {render_mode}"

        
        self.actions_noop_count = 0
        self.actions_direct_count = 0
        self.actions_steer_count = 0
        self.last_continuous_action = 0.0
        

        self.action_markers = {}

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
        
    def _get_info(self): 
        return {**super()._get_info(),
        'actions_noop': int(self.actions_noop_count),
        'actions_direct': int(self.actions_direct_count),
        'actions_steer': int(self.actions_steer_count),
        'last_continuous_action': float(self.last_continuous_action),
        }
        
    def reset(self, seed=None, options=None):
        self.actions_noop_count = 0
        self.actions_direct_count = 0
        self.actions_steer_count = 0

        return super().reset(seed, options)
    
    def step(self, action):
        aktiv_agent = self.all_agents.get_active_agent()
        self._set_action(action, aktiv_agent)
        return BaseCrossingEnv._step(self, action)
    
    def _gen_aircraft(self, num_episodes):
        return BaseCrossingEnv._gen_aircraft(self, num_episodes)

    def _get_observation(self, agent: Optional[Agent] = None) -> np.ndarray:
        return BaseCrossingEnv._get_observation(self, agent)
    
    def _get_reward(self) -> float:
        return BaseCrossingEnv._get_reward(self)

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
                self.actions_direct_count += 1
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
        BaseCrossingEnv.render(self, self.render_mode)
        
    def close(self):
        BaseCrossingEnv.close(self)

    
