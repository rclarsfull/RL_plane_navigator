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

from .helper_classes import Waypoint, Agent, Agents
from .consts import *
from .helper_functions import bound_angle_positive_negative_180
from .base_crossing_env import BaseCrossingEnv

logger = logging.getLogger(__name__)


class Cross_env(BaseCrossingEnv, gym.Env):
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
        
        self.step_limit = int(TIME_LIMIT / AGENT_INTERACTION_TIME)
        self.resets_counter = 0

        self.steps = 0
        self.actions_noop_count = 0
        self.actions_steer_count = 0
        self.last_continuous_action = 0.0
        
        self.action_markers = {}

        self.num_steer_bins = 37
        # New simple discrete action space:
        # 0 = NOOP, 1 = DIRECT (snap to next waypoint), 2..(2+num_steer_bins-1) = steer bins
        self.action_space = gym.spaces.Discrete(self.num_steer_bins + 2)
        # Human readable mapping for debugging / rendering
        self.action_names = ["noop", "direct"] + [f"steer_{i}" for i in range(self.num_steer_bins)]
        # Counters
        self.actions_direct_count = 0
        # Erzeuge exponentiell verteilte Winkel-Bins (symmetrisch um 0°)
        self.steer_angle_bins = self._build_steer_angle_bins(alpha=3.0)

        self.total_reward = 0.0
        self.num_episodes = 0
        self.total_intrusions = 0
        
    def step(self, action):
        """Wrapper - uses base class implementation"""
        aktiv_agent = self.all_agents.get_active_agent()
        self._set_action(action, aktiv_agent)
        return BaseCrossingEnv._step(self, action)
    
    def _gen_aircraft(self, num_episodes):
        """Wrapper method that calls base class implementation

        Accepts `num_episodes` because `BaseCrossingEnv.reset` calls
        `self._gen_aircraft(self.num_episodes)`.
        """
        return BaseCrossingEnv._gen_aircraft(self, num_episodes)

    def _get_observation(self, agent: Optional[Agent] = None) -> np.ndarray:
        """Wrapper - uses base class implementation"""
        return BaseCrossingEnv._get_observation(self, agent)
    
    def _get_info(self):
        """Extends base class info with action statistics"""
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
        return BaseCrossingEnv.reset(self,seed=seed, options=options)
    
    def _get_reward(self) -> float:
        """Wrapper - uses base class implementation"""
        return BaseCrossingEnv._get_reward(self)


    def _set_action(self, action, agent: Agent) -> None:
        """Set action using the new Discrete action encoding.

        Encoding:
            0 -> NOOP
            1 -> DIRECT (snap to next waypoint)
            2.. -> STEER (index = action - 2)
        """
        BaseCrossingEnv._set_action(self, action, agent)
        # action is expected to be an integer in the new discrete space
        try:
            a = int(action)
        except Exception:
            # Fallback: treat non-int as noop
            a = 0

        action_type = a  # 0=NOOP,1=DIRECT,>=2=STEER
        steering_index = a - 2  # only used when >=2

        is_active_agent = (agent == self.all_agents.get_active_agent())

        # Für Render Logging: Hole aktuelle Position (falls nötig)
        if self.render_mode is not None:
            lat, lon, _, _ = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(self.sim, lat, lon, self.window_width, self.window_height)
        else:
            x_pos, y_pos = None, None

        if action_type == 0:  # NOOP
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

        # DIRECT (snap) oder STEER Aktion
        agent.is_noop = False
        agent.last_action = 1
        if is_active_agent:
            if action_type == 1:
                # direct course
                self.actions_direct_count += 1
            elif action_type == 2:
                self.actions_steer_count += 1

        _, _, current_heading, _ = self.sim.traf_get_state(agent.id)

        if action_type == 1:  # DIRECT / SNAP
            waypoint = self.getNextWaypoint(agent)
            target_dir, _ = self.sim.geo_calculate_direction_and_distance(agent.ac_lat, agent.ac_lon, waypoint.lat, waypoint.lon)
            heading_new = bound_angle_positive_negative_180(target_dir)
            rel_delta = bound_angle_positive_negative_180(heading_new - current_heading)
            continuous_steering = float(np.clip(rel_delta / 180.0, -1.0, 1.0))
            agent.last_action_type = 'direct'
        elif action_type == 2:  # STEER (nichtlineare Winkel aus Lookup-Tabelle)
            if steering_index < 0 or steering_index >= self.num_steer_bins:
                rel_deg = 0.0
            else:
                rel_deg = float(self.steer_angle_bins[steering_index])
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

    def _build_steer_angle_bins(self, alpha: float = 3.0) -> np.ndarray:
        """Erzeugt eine symmetrische Liste relativer Kursänderungen in Grad mit
        exponentieller Verdichtung nahe 0°. alpha steuert die Krümmung:
        Höher => stärkere Verdichtung um 0.

        Index-Belegung:
            0 .......... -> -max_angle
            mid ........ -> 0°
            last ........ -> +max_angle

        Alle Zwischenwerte sind streng monoton steigend.
        """
        max_angle = 180.0
        n = self.num_steer_bins
        mid = n // 2  # enthält 0° bei ungerader Anzahl
        angles = []
        exp_max = np.exp(alpha) - 1.0
        for i in range(n):
            if i == mid:
                angles.append(0.0)
                continue
            # Normalisiere Index auf [-1,1]
            x = (i - mid) / mid  # -1 .. 1
            mag = (np.exp(alpha * abs(x)) - 1.0) / exp_max  # 0 .. 1
            angle = mag * max_angle * np.sign(x)
            # Runde auf sinnvolle Grade (optional). Hier keine Rundung für feinere Differenzierung.
            angles.append(angle)
        return np.array(angles, dtype=np.float32)
