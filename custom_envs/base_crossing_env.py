"""
BASE CLASS für alle Crossing Planes Environments

Enthält alle gemeinsamen Methoden:
- CPA-Berechnung und Collision Detection
- Observation Space Generation
- Aircraft Generation
- Reward Computation (basis)
- Rendering Logic
"""

import logging
import random
import time
import numpy as np
import pygame
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from numba import njit


import bluesky_gym.envs.common.functions as fn

from .helper_classes import Waypoint, Agent, Agents, Camera
from .consts import *
from .helper_functions import bound_angle_positive_negative_180, get_point_at_distance, compute_cpa_multi_heading_numba
from simulator.blue_sky_adapter import Simulator

logger = logging.getLogger(__name__)

class BaseCrossingEnv:
    """Base class with common methods for all crossing environments"""
    
    def __init__(self, center_lat: float, center_lon: float, render_mode: Optional[str] = None, 
                 window_width: int = 1500, window_height: int = 1000):
        # Initialize simulator
        self.sim = Simulator()
        self.sim.init(simulator_step_size=1.0)
        
        # Set environment center
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.all_agents: Optional[Agents] = None
        
        # Parser cache - load once on initialization
        self.parser_cache = None
        self.parser_cache_path = None
        
        # Rendering setup
        self.render_mode = render_mode
        self.window_width = window_width
        self.window_height = window_height
        self.window_size = (window_width, window_height)
        self.window = None
        self.clock = None
        self.is_paused = False
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        
        # Camera and trails
        self.agent_trails = {}
        self.camera = Camera(center_lat=center_lat, center_lon=center_lon, zoom_km=667)
    
    def _load_parser_data(self, json_path: str):
        """Load parser data once and cache it"""
        if self.parser_cache is None or self.parser_cache_path != json_path:
            logger.debug(f"Loading parser data from {json_path} (this happens only once)")
            try:
                from parser.parse_flightplans import parse_all_exercises
                self.parser_cache = parse_all_exercises(json_path)
                self.parser_cache_path = json_path
                logger.debug(f"Cached {len(self.parser_cache)} exercises from {json_path}")
            except Exception as e:
                logger.error(f"Failed to load parser data: {e}")
                self.parser_cache = None
                raise
        return self.parser_cache
    
    # ==================== COLLISION DETECTION ====================
    
    def _is_actual_danger(self, closing_rate, time_to_min_sep, min_separation):
        """Check if situation is actually dangerous (vectorized)"""
        is_closing = closing_rate < DANGER_CLOSING_THRESHOLD
        is_near_future = (time_to_min_sep > 0) & (time_to_min_sep < LONG_CONFLICT_THRESHOLD_SEC)
        is_dangerous = min_separation < DANGER_MIN_SEP_THRESHOLD
        return is_closing & is_near_future & is_dangerous
    
    
    def _calculate_collision_criticality_vectorized(
        self, agent_lat, agent_lon, agent_hdg, agent_tas,
        intruder_lats, intruder_lons, intruder_hdgs, intruder_tas_array
    ):
        n_intruders = len(intruder_lats)
        
        if n_intruders == 0:
            return {}
        
        bearings = np.zeros(n_intruders, dtype=np.float64)
        distances = np.zeros(n_intruders, dtype=np.float64)
        
        for i in range(n_intruders):
            bearings[i], distances[i] = self.sim.geo_calculate_direction_and_distance(
                agent_lat, agent_lon, intruder_lats[i], intruder_lons[i]
            )
        
        min_seps_2d, t_to_cpas_2d, c_rates_2d = compute_cpa_multi_heading_numba(
            agent_hdg, agent_tas,
            intruder_hdgs, intruder_tas_array,
            bearings, distances,
            np.array([0.0], dtype=np.float64)
        )
        
        min_separation = min_seps_2d[:, 0]
        t_to_cpa = t_to_cpas_2d[:, 0]
        closing_rate = c_rates_2d[:, 0]
        
        is_colliding = distances <= INTRUSION_DISTANCE
        flugbahnen_schneiden = min_separation <= INTRUSION_DISTANCE
        future_collision = (min_separation <= INTRUSION_DISTANCE * 1.5) & (closing_rate < 0) & (t_to_cpa > 0) & (t_to_cpa < 30.0)
        
        results = {}
        for i in range(n_intruders):
            results[i] = {
                'min_separation': float(min_separation[i]),
                'time_to_min_sep': float(t_to_cpa[i]),
                'closing_rate': float(closing_rate[i]),
                'is_colliding': bool(is_colliding[i]),
                'flugbahnen_schneiden': bool(flugbahnen_schneiden[i]),
                'future_collision': bool(future_collision[i])
            }
        
        del bearings, distances, min_separation, t_to_cpa, closing_rate, is_colliding, flugbahnen_schneiden, future_collision
        return results

    
    def _compute_multi_heading_cpa_features(
        self, agent: Agent, ac_lat: float, ac_lon: float, ac_hdg: float, ac_tas: float
    ) -> np.ndarray:
        """Compute multi-heading CPA features (vectorized)

        Rückgabeformat: Länge = NUM_HEADING_OFFSETS + 1
        - Indizes 0..(NUM_HEADING_OFFSETS-1): fixe Offsets gemäß HEADING_OFFSETS
        - Letzter Index: hypothetische Ausrichtung direkt zum Ziel-Heading (Bearing zum nächsten Waypoint)
        Wertebereich jeweils in [0,1]: größer = weniger kritisch (länger bis zur minimalen Separation)
        """
        features_len = NUM_HEADING_OFFSETS + 1
        multi_heading_cpa_features = np.ones(features_len, dtype=np.float32)
        
        if not hasattr(agent, 'intruder_state_map') or len(agent.intruder_state_map) == 0:
            return multi_heading_cpa_features
        
        intruder_data = list(agent.intruder_state_map.values())
        if len(intruder_data) == 0:
            return multi_heading_cpa_features
        
        intruder_lats = np.array([d[0] for d in intruder_data], dtype=np.float64)
        intruder_lons = np.array([d[1] for d in intruder_data], dtype=np.float64)
        intruder_hdgs = np.array([d[2] for d in intruder_data], dtype=np.float64)
        intruder_tas_array = np.array([d[3] for d in intruder_data], dtype=np.float64)
        
        n_intruders = len(intruder_lats)
        
        bearings = np.zeros(n_intruders, dtype=np.float64)
        distances = np.zeros(n_intruders, dtype=np.float64)
        for i in range(n_intruders):
            bearings[i], distances[i] = self.sim.geo_calculate_direction_and_distance(
                ac_lat, ac_lon, intruder_lats[i], intruder_lons[i]
            )
        
        # Erzeuge Offsets-Array: fixe Offsets + (optional) Ziel-Heading in einem Durchlauf
        include_goal = True
        try:
            waypoint = self.getNextWaypoint(agent)
            target_dir_deg, _ = self.sim.geo_calculate_direction_and_distance(ac_lat, ac_lon, waypoint.lat, waypoint.lon)
            target_offset = bound_angle_positive_negative_180(target_dir_deg - ac_hdg)
            offsets_combined = np.concatenate([HEADING_OFFSETS, np.array([target_offset], dtype=np.float64)])
        except Exception as e:
            logger.debug(f"Goal-CPA bearing computation failed, skipping goal offset: {e}")
            offsets_combined = HEADING_OFFSETS
            include_goal = False

        min_seps, t_cpas, c_rates = compute_cpa_multi_heading_numba(
            ac_hdg, ac_tas,
            intruder_hdgs, intruder_tas_array,
            bearings, distances,
            offsets_combined
        )

        is_critical = self._is_actual_danger(c_rates, t_cpas, min_seps)
        t_cpas_masked = np.where(is_critical, t_cpas, np.inf)
        min_critical_times = np.min(t_cpas_masked, axis=0)

        norm_vals = np.ones(offsets_combined.shape[0], dtype=np.float32)
        sel_mask = np.isfinite(min_critical_times)
        norm_vals[sel_mask] = np.clip(min_critical_times[sel_mask] / LONG_CONFLICT_THRESHOLD_SEC, 0.0, 1.0)

        # Schreibe Basis-Offsets
        base_count = NUM_HEADING_OFFSETS
        multi_heading_cpa_features[:base_count] = norm_vals[:base_count]
        # Schreibe ggf. Ziel-Heading-Wert an letzte Position
        if include_goal and norm_vals.shape[0] > base_count:
            multi_heading_cpa_features[base_count] = norm_vals[base_count]
        
        del bearings, distances, min_seps, t_cpas, c_rates, is_critical, t_cpas_masked
        return multi_heading_cpa_features
    
    def _compute_action_history(self, agent: Agent) -> np.ndarray:
        """Compute action history features for observation"""
        return np.array([
            agent.last_action, 
            agent.last_action_continuous, 
            agent.action_age,
            agent.turning_rate
        ], dtype=np.float32)
    
    def _select_intruders_by_cpa(
        self, agent: Agent, ac_lat: float, ac_lon: float, ac_hdg: float, ac_tas: float, collision_info_cache: Dict
    ) -> List[str]:
        """Select most dangerous intruders by CPA"""
        real_threats = []
        
        for intruder_id, collision_info in collision_info_cache.items():
            min_separation = collision_info['min_separation']
            closing_rate = collision_info['closing_rate']
            time_to_min_sep = collision_info['time_to_min_sep']
            
            if self._is_actual_danger(closing_rate, time_to_min_sep, min_separation):
                real_threats.append((intruder_id, time_to_min_sep))
        
        real_threats.sort(key=lambda x: x[1])
        selected_ids_cpa = [intruder_id for intruder_id, _ in real_threats[:NUM_AC_STATE]]
        
        return selected_ids_cpa
    
    def _compute_intruder_features(
        self, agent: Agent, ac_lat: float, ac_lon: float, ac_hdg: float, ac_tas: float
    ) -> np.ndarray:
        """Compute intruder features for observation"""
        FILLER_SLOT = [1.0, -1.0, 0.0, 0.0, 1.0, 1.0, ac_tas, 0.0, 0.0]
        intruder_features = np.array(FILLER_SLOT * NUM_AC_STATE, dtype=np.float32)
        
        closest_ids = agent.selected_intruder_ids
        
        for idx, intruder_id in enumerate(closest_ids):
            if intruder_id not in agent.intruder_state_map:
                continue
            
            int_lat, int_lon, int_hdg, int_tas = agent.intruder_state_map[intruder_id]
            int_rel_lat = int_lat - ac_lat
            int_rel_lon = int_lon - ac_lon
            collision_info = agent.intruder_collision_cache[intruder_id]
            distance_nm = agent.intruder_distance_map[intruder_id]
            
            
            bearing, _ = self.sim.geo_calculate_direction_and_distance(ac_lat, ac_lon, int_lat, int_lon)
            bearing_diff = bound_angle_positive_negative_180(bearing - ac_hdg)
            bearing_diff_rad = np.deg2rad(bearing_diff)
            bearing_cos = np.cos(bearing_diff_rad)
            bearing_sin = np.sin(bearing_diff_rad)
            
            closing_rate = collision_info['closing_rate']
            
            min_separation = collision_info['min_separation']
            
            time_to_min_sep = collision_info['time_to_min_sep']
            
            cpa_lat_cached = None
            cpa_lon_cached = None
            
            if time_to_min_sep > 0 and time_to_min_sep < LONG_CONFLICT_THRESHOLD_SEC:
                try:
                    active_vel_nm_s = ac_tas / 1852.0
                    active_distance_traveled = active_vel_nm_s * time_to_min_sep
                    agent_cpa_lat, agent_cpa_lon = get_point_at_distance(
                        ac_lat, ac_lon, active_distance_traveled, ac_hdg
                    )
                    
                    intruder_vel_nm_s = int_tas / 1852.0
                    intruder_distance_traveled = intruder_vel_nm_s * time_to_min_sep
                    intruder_cpa_lat, intruder_cpa_lon = get_point_at_distance(
                        int_lat, int_lon, intruder_distance_traveled, int_hdg
                    )
                    
                    cpa_lat_cached = (agent_cpa_lat + intruder_cpa_lat) / 2.0
                    cpa_lon_cached = (agent_cpa_lon + intruder_cpa_lon) / 2.0
                    
                    bearing_to_cpa, distance_to_cpa = self.sim.geo_calculate_direction_and_distance(
                        ac_lat, ac_lon, cpa_lat_cached, cpa_lon_cached
                    )
                    
                    bearing_to_cpa_rel = bound_angle_positive_negative_180(bearing_to_cpa - ac_hdg)
                    bearing_to_cpa_rad = np.deg2rad(bearing_to_cpa_rel)
                    
                    
                except Exception as e:
                    logger.debug(f"Error computing CPA position for {intruder_id}: {e}")
            
            if not hasattr(agent, 'cpa_position_map'):
                agent.cpa_position_map = {}
            agent.cpa_position_map[intruder_id] = (cpa_lat_cached, cpa_lon_cached)
            
            features_list = [
                distance_nm,
                bearing_cos, bearing_sin,
                closing_rate,
                min_separation,
                time_to_min_sep,
                int_tas,
                int_rel_lat, int_rel_lon
            ]
            
            base_idx = idx * 9
            intruder_features[base_idx:base_idx+9] = features_list
        
        return intruder_features
    
    # ==================== OBSERVATION GENERATION ====================
    
    def _get_observation_dict(self, agent: Optional[Agent] = None) -> Dict[str, np.ndarray]:
        """
        Get observation dictionary with all components
        
        Returns dict with:
        - ego_state: (6) Heading, Speed, Drift, V-Sep, Distance to Waypoint
        - threat_features: (NUM_AC_STATE*9) Intruder Features (9 per intruder)
        - action_history: (5) Last Action, Age, Turn Rate, Speed Action
        - multi_heading_cpa: (NUM_HEADING_OFFSETS) Min time_to_cpa for each heading offset
        """
        if agent is None:
            agent = self.all_agents.get_active_agent()
        
        ac_lat, ac_lon, ac_hdg, ac_tas = agent.ac_lat, agent.ac_lon, agent.ac_hdg, agent.ac_tas
        
        # Ego State
        ac_hdg_rad = np.deg2rad(ac_hdg)
        ego_hdg_cos = np.cos(ac_hdg_rad)
        ego_hdg_sin = np.sin(ac_hdg_rad)
        
        speed_normalized = agent.speed_normalized
        drift_normalized = agent.drift_normalized
        vertical_separation_normalized = 1.0
        distance_to_waypoint_normalized = agent.distance_to_waypoint_normalized
        
        # Features
        ego_state = np.array([
            ego_hdg_cos, ego_hdg_sin,
            speed_normalized,
            drift_normalized,
            vertical_separation_normalized,
            distance_to_waypoint_normalized
        ], dtype=np.float32)
        threat_features = self._compute_intruder_features(agent, ac_lat, ac_lon, ac_hdg, ac_tas)
        multi_heading_cpa = self._compute_multi_heading_cpa_features(agent, ac_lat, ac_lon, ac_hdg, ac_tas)
        action_history = self._compute_action_history(agent)
        
        return {
            'ego_state': ego_state,
            'threat_features': threat_features,
            'action_history': action_history,
            'multi_heading_cpa': multi_heading_cpa
        }
    
    def _step(self, action):
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
    
    def _get_observation(self, agent: Optional[Agent] = None) -> np.ndarray:
        """Get flat observation"""
        if agent is None:
            agent = self.all_agents.get_active_agent()
        
        obs_dict = agent.obs_dict_cache if agent.obs_dict_cache is not None else self._get_observation_dict(agent)
        
        flat_obs = np.concatenate([
            obs_dict['ego_state'],
            obs_dict['threat_features'],
            obs_dict['action_history'],
            obs_dict['multi_heading_cpa']
        ])
        return flat_obs
    
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
    
    # ==================== AIRCRAFT GENERATION ====================
    
    def _create_agents(self, n_agents, n_random_agents):
        """Create agent list"""
        agents = Agents()
        total_agents = n_agents + n_random_agents
        for i in range(total_agents):
            agents.add_agent(Agent(f'kl00{i+1}'.upper(), 0.0, False, 999.0, 0.0))
        return agents
    
    def _add_random_agents(self, routes_dict, center_lat, center_lon, n_agents, n_random_agents):
        """Helper to add random agents to the scenario"""
        outer_ring_radius_nm = 200
        outer_ring_radius_km = outer_ring_radius_nm * NM2KM
        
        for random_agent_idx in range(n_random_agents):
            agent_idx = n_agents + random_agent_idx
            
            offset_x_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
            offset_y_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
            start_lat = center_lat + (offset_x_km / 110.54)
            start_lon = center_lon + (offset_y_km / 111.32)
            
            waypoints = deque()
            
            
            for i in range(5):
                offset_x_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
                offset_y_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
                wp_lat = center_lat + (offset_x_km / 110.54)
                wp_lon = center_lon + (offset_y_km / 111.32)
                waypoints.append(Waypoint(wp_lat, wp_lon))
            
            routes_dict[agent_idx] = {
                'waypoints': waypoints,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'start_heading': random.uniform(0, 360),
                'speed': random.uniform(MIN_SPEED, MAX_SPEED),
                'ring_idx': -1
            }
        return routes_dict

    def _add_random_agents_at_flight_level(self, routes_dict, center_lat, center_lon, n_agents, n_random_agents, flight_level):
        """Helper to add random agents to the scenario at a specific flight level"""
        outer_ring_radius_nm = 200
        outer_ring_radius_km = outer_ring_radius_nm * NM2KM
        
        for random_agent_idx in range(n_random_agents):
            agent_idx = n_agents + random_agent_idx
            
            offset_x_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
            offset_y_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
            start_lat = center_lat + (offset_x_km / 110.54)
            start_lon = center_lon + (offset_y_km / 111.32)
            
            waypoints = deque()
            
            for i in range(5):
                offset_x_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
                offset_y_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
                wp_lat = center_lat + (offset_x_km / 110.54)
                wp_lon = center_lon + (offset_y_km / 111.32)
                waypoints.append(Waypoint(wp_lat, wp_lon))
            
            routes_dict[agent_idx] = {
                'waypoints': waypoints,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'start_heading': random.uniform(0, 360),
                'speed': random.uniform(MIN_SPEED, MAX_SPEED),
                'altitude': flight_level
            }
        return routes_dict

    def _generate_crossing_scenario(self, center_lat, center_lon, n_agents, n_random_agents):
        """Generate crossing routes with ring distribution (Scenario 1)"""
        routes_dict = {}
        common_speed = random.uniform(MIN_SPEED, MAX_SPEED)
        
        ring_assignments = [(i, random.randint(0, 3)) for i in range(n_agents)]
        angles_by_ring = {0: [], 1: [], 2: [], 3: []}
        max_tries = 200
        
        for ring_idx in range(4):
            agents_in_ring = len([a for a in ring_assignments if a[1] == ring_idx])
            if agents_in_ring == 0:
                continue
            
            angle_spacing = 360.0 / agents_in_ring if agents_in_ring > 1 else 180.0
            
            for _ in range(agents_in_ring):
                found = False
                for _ in range(max_tries):
                    candidate_angle = random.uniform(0, 360)
                    valid = True
                    for existing_angle in angles_by_ring[ring_idx]:
                        if abs(candidate_angle - existing_angle) < angle_spacing * 0.5:
                            valid = False
                            break
                    if valid:
                        angles_by_ring[ring_idx].append(candidate_angle)
                        found = True
                        break
                
                if not found:
                    base_angle = (360.0 / agents_in_ring) * len(angles_by_ring[ring_idx])
                    jitter = random.uniform(-10, 10)
                    angles_by_ring[ring_idx].append((base_angle + jitter) % 360)
        
        angle_index = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for agent_idx, (_, assigned_ring) in enumerate(ring_assignments):
            ring_radius_nm = 100 + (assigned_ring * 15)
            ring_radius_km = ring_radius_nm * NM2KM
            
            angle_deg = angles_by_ring[assigned_ring][angle_index[assigned_ring]]
            angle_index[assigned_ring] += 1
            angle_rad = np.deg2rad(angle_deg)
            
            start_x_km = ring_radius_km * np.cos(angle_rad)
            start_y_km = ring_radius_km * np.sin(angle_rad)
            start_lat = center_lat + (start_x_km / 110.54)
            start_lon = center_lon + (start_y_km / 111.32)
            
            conflict_offset_nm = random.uniform(-8.0, 8.0)
            conflict_offset_km = conflict_offset_nm * NM2KM
            
            waypoints = deque()
            
            if random.random() < 0.2:
                waypoints.append(Waypoint(center_lat, center_lon))
                angle_deg = angle_deg + random.uniform(-15, 15)
            
            exit_angle_deg = angle_deg + 180.0
            exit_angle_rad = np.deg2rad(exit_angle_deg)
            exit_x_km = ring_radius_km * np.cos(exit_angle_rad)
            exit_y_km = ring_radius_km * np.sin(exit_angle_rad)
            
            offset_perp_angle_rad = exit_angle_rad + np.pi / 2
            exit_x_km += conflict_offset_km * np.cos(offset_perp_angle_rad)
            exit_y_km += conflict_offset_km * np.sin(offset_perp_angle_rad)
            
            exit_lat = center_lat + (exit_x_km / 110.54)
            exit_lon = center_lon + (exit_y_km / 111.32)
            waypoints.append(Waypoint(exit_lat, exit_lon))
            
            first_waypoint = waypoints[0]
            start_heading, _ = self.sim.geo_calculate_direction_and_distance(
                start_lat, start_lon, first_waypoint.lat, first_waypoint.lon
            )
            
            routes_dict[agent_idx] = {
                'waypoints': waypoints,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'start_heading': start_heading,
                'speed': common_speed
            }
        
        return self._add_random_agents(routes_dict, center_lat, center_lon, n_agents, n_random_agents)

    def _generate_merging_scenario(self, center_lat, center_lon, n_agents, n_random_agents):
        """Generate merging scenario: Agents converge to a common waypoint (Scenario 2)"""
        routes_dict = {}
        common_speed = random.uniform(MIN_SPEED, MAX_SPEED)
        
        # Common exit point for all merging agents
        exit_angle_deg = random.uniform(0, 360)
        exit_angle_rad = np.deg2rad(exit_angle_deg)
        exit_dist_km = 150.0
        
        exit_x_km = exit_dist_km * np.cos(exit_angle_rad)
        exit_y_km = exit_dist_km * np.sin(exit_angle_rad)
        exit_lat = center_lat + (exit_x_km / 110.54)
        exit_lon = center_lon + (exit_y_km / 111.32)
        
        # Agents start in a sector "behind" the merge point
        reciprocal_angle = (exit_angle_deg + 180) % 360
        start_dist_km = 150.0
        
        # Distribute agents
        angle_spread = 120.0 # degrees
        if n_agents > 1:
            angle_step = angle_spread / (n_agents - 1)
            start_angles = [reciprocal_angle - (angle_spread/2) + i*angle_step for i in range(n_agents)]
        else:
            start_angles = [reciprocal_angle]
            
        for i, angle_deg in enumerate(start_angles):
            angle_rad = np.deg2rad(angle_deg)
            start_x_km = start_dist_km * np.cos(angle_rad)
            start_y_km = start_dist_km * np.sin(angle_rad)
            
            start_lat = center_lat + (start_x_km / 110.54)
            start_lon = center_lon + (start_y_km / 111.32)
            
            waypoints = deque()
            # Merge point
            waypoints.append(Waypoint(center_lat, center_lon))
            # Exit point
            waypoints.append(Waypoint(exit_lat, exit_lon))
            
            start_heading, _ = self.sim.geo_calculate_direction_and_distance(
                start_lat, start_lon, center_lat, center_lon
            )
            
            routes_dict[i] = {
                'waypoints': waypoints,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'start_heading': start_heading,
                'speed': common_speed
            }
            
        return self._add_random_agents(routes_dict, center_lat, center_lon, n_agents, n_random_agents)

    def _generate_diverging_scenario(self, center_lat, center_lon, n_agents, n_random_agents):
        """Generate diverging scenario: Agents start together and split (Scenario 3)"""
        routes_dict = {}
        common_speed = random.uniform(MIN_SPEED, MAX_SPEED)
        
        # Common start direction (incoming)
        incoming_angle_deg = random.uniform(0, 360)
        incoming_angle_rad = np.deg2rad(incoming_angle_deg)
        start_dist_km = 150.0
        
        # Agents are spaced out along the incoming route
        spacing_km = 10.0 * NM2KM
        
        # Exit angles spread out "forward"
        exit_base_angle = (incoming_angle_deg + 180) % 360
        angle_spread = 120.0
        
        if n_agents > 1:
            angle_step = angle_spread / (n_agents - 1)
            exit_angles = [exit_base_angle - (angle_spread/2) + i*angle_step for i in range(n_agents)]
        else:
            exit_angles = [exit_base_angle]
        
        # Randomize agent order so active agent isn't always first
        position_order = list(range(n_agents))
        random.shuffle(position_order)
            
        for agent_idx in range(n_agents):
            position = position_order[agent_idx]
            dist_from_center = start_dist_km + (position * spacing_km)
            s_x = dist_from_center * np.cos(incoming_angle_rad)
            s_y = dist_from_center * np.sin(incoming_angle_rad)
            
            start_lat = center_lat + (s_x / 110.54)
            start_lon = center_lon + (s_y / 111.32)
            
            waypoints = deque()
            # Diverge point (Center)
            waypoints.append(Waypoint(center_lat, center_lon))
            
            # Individual Exit Point
            ex_angle_deg = exit_angles[agent_idx]
            ex_angle_rad = np.deg2rad(ex_angle_deg)
            ex_dist_km = 150.0
            
            ex_x = ex_dist_km * np.cos(ex_angle_rad)
            ex_y = ex_dist_km * np.sin(ex_angle_rad)
            
            exit_lat = center_lat + (ex_x / 110.54)
            exit_lon = center_lon + (ex_y / 111.32)
            
            waypoints.append(Waypoint(exit_lat, exit_lon))
            
            start_heading, _ = self.sim.geo_calculate_direction_and_distance(
                start_lat, start_lon, center_lat, center_lon
            )
            
            routes_dict[agent_idx] = {
                'waypoints': waypoints,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'start_heading': start_heading,
                'speed': common_speed
            }
            
        return self._add_random_agents(routes_dict, center_lat, center_lon, n_agents, n_random_agents)

    def _generate_mixed_scenario(self, center_lat, center_lon, n_agents, n_random_agents, 
                                json_path: str = "parser/scenarios/ACS-exercises.json"):
        """Select one of the 4 scenarios based on probabilities: 3 random + 1 parser"""
        scenario_type = random.choices(
            ['crossing', 'merging', 'diverging', 'parser'],
            weights=[0.50, 0.10, 0.02, 0.05],  # Adjusted to include parser (28%)
            k=1
        )[0]
        
        logger.debug(f"Generating scenario: {scenario_type}")
        
        if scenario_type == 'crossing':
            return self._generate_crossing_scenario(center_lat, center_lon, n_agents, n_random_agents)
        elif scenario_type == 'merging':
            return self._generate_merging_scenario(center_lat, center_lon, n_agents, n_random_agents)
        elif scenario_type == 'diverging':
            return self._generate_diverging_scenario(center_lat, center_lon, n_agents, n_random_agents)
        else:  # parser
            return self._generate_parser_scenario(
                center_lat=center_lat,
                center_lon=center_lon,
                n_random_agents=n_random_agents,
                json_path=json_path,
                flight_level=None,
                speed_override=None,
                exercise_id=None
            )
    
    def _generate_parser_scenario(self, center_lat, center_lon, n_random_agents=0, 
                                  json_path: str = "parser/scenarios/ACS-exercises.json",
                                  flight_level: Optional[int] = None,
                                  speed_override: Optional[float] = None,
                                  exercise_id: Optional[int] = None,
                                  time_window_seconds: int = 3600):
        """
        Generate scenario by loading flight plans from parser/exercises.json
        
        Args:
            center_lat: Center latitude for scenario
            center_lon: Center longitude for scenario
            n_random_agents: Number of random intruder agents to add
            json_path: Path to exercises.json file
            flight_level: Optional altitude override (in hundreds of feet). 
                         If None, a flight level from 2-3 quartile is selected
            speed_override: Optional speed override (in m/s).
                           If None, uses TAS from flightplan data
            exercise_id: Optional specific Exercise ID. If None, a random exercise is selected
            time_window_seconds: Duration of the time window to filter flightplans (default 3600 = 1 hour)
        
        Returns:
            routes_dict with agent routes
        """
        routes_dict = {}
        
        # Load flightplan data for the exercise
        logger.debug(f"Loading {'specific' if exercise_id else 'random'} exercise from {json_path}")
        
        try:
            if exercise_id is not None:
                from parser.parse_flightplans import get_specific_exercise
                loaded_exercise_id, flightplans = get_specific_exercise(json_path, exercise_id)
            else:
                from parser.parse_flightplans import get_random_exercise
                loaded_exercise_id, flightplans = get_random_exercise(json_path)
        except Exception as e:
            logger.error(f"Failed to load exercise: {e}")
            raise
        
        if not flightplans:
            logger.warning(f"No waypoints found for exercise {loaded_exercise_id}")
            raise ValueError(f"No flightplans for exercise {loaded_exercise_id}")
        
        logger.debug(f"Loaded Exercise #{loaded_exercise_id} with {len(flightplans)} flightplans")
        
        # Filter flightplans by time window AND flight level
        from parser.parse_flightplans import filter_flightplans_by_time_window_and_flight_level
        flightplans, window_start, window_end, selected_flight_level = filter_flightplans_by_time_window_and_flight_level(
            flightplans, 
            window_duration_seconds=time_window_seconds,
            target_flight_level=flight_level  # Use provided flight_level or select from quartile
        )
        
        logger.debug(f"Filtered to {len(flightplans)} flightplans in time window ETO {window_start}-{window_end}s, Flight Level {selected_flight_level}")
        
        if not flightplans:
            logger.warning(f"No flightplans in time window/flight level for exercise {loaded_exercise_id}")
            raise ValueError(f"No flightplans in time window/flight level for exercise {loaded_exercise_id}")
        
        # Create agents from flightplan data
        agent_idx = 0
        for fp_id, fp_waypoints in flightplans.items():
            if not fp_waypoints:
                continue
            
            # fp_waypoints is a list of waypoint dicts
            first_waypoint_data = fp_waypoints[0]
            
            # Get speed
            speed = speed_override if speed_override is not None else first_waypoint_data.get('tas', MIN_SPEED)
            # Convert TAS (in knots) to m/s if needed
            if speed > 400:  # Assume it's in knots if too high
                speed = speed * 0.51444  # knots to m/s
            speed = np.clip(speed, MIN_SPEED, MAX_SPEED)
            
            # Use the selected flight level for all aircraft
            altitude = selected_flight_level
            
            # Create waypoint queue from all waypoints
            wp_queue = deque()
            for wp_data in fp_waypoints:
                wp_queue.append(Waypoint(wp_data['lat'], wp_data['lon']))
            
            # Get first waypoint for start position
            first_wp = wp_queue[0]
            
            # Calculate heading from first to second waypoint
            start_heading = 0.0
            if len(wp_queue) > 1:
                second_wp = wp_queue[1]
                start_heading, _ = self.sim.geo_calculate_direction_and_distance(
                    first_wp.lat, first_wp.lon, second_wp.lat, second_wp.lon
                )
            
            aircraft_type = first_waypoint_data.get('aircraft_type', 'A320')
            
            routes_dict[agent_idx] = {
                'waypoints': wp_queue,
                'start_lat': first_wp.lat,
                'start_lon': first_wp.lon,
                'start_heading': start_heading,
                'speed': speed,
                'ring_idx': 0,
                'altitude': altitude,
                'flightplan_id': fp_id,
                'aircraft_type': aircraft_type,
                'eto_start': first_waypoint_data.get('eto', 0)
            }
            
            logger.debug(f"Agent {agent_idx}: FP#{fp_id}, Speed={speed:.1f}m/s, Alt={altitude}, Aircraft={aircraft_type}, ETO={first_waypoint_data.get('eto', 0)}")
            agent_idx += 1
        
        n_agents = agent_idx
        
        # Add random intruders if requested (also on same flight level)
        if n_random_agents > 0:
            routes_dict = self._add_random_agents_at_flight_level(
                routes_dict, center_lat, center_lon, n_agents, n_random_agents, selected_flight_level
            )
        
        logger.debug(f"Parser scenario: Exercise#{loaded_exercise_id}, {n_agents} planned agents + {n_random_agents} random agents, Flight Level {selected_flight_level}")
        return routes_dict
    
    def _gen_aircraft(self, num_episodes: int,
                      parser_path: str = "parser/scenarios/ACS-exercises.json"):
        """
        Generate aircraft for the episode.
        Configuration of agent counts happens here.
        
        Args:
            num_episodes: Episode number for logging
            parser_path: Path to exercises.json for parser scenario
        """
        # Configure scenario parameters here
        n_agents = random.randint(2, 4)          # Planned agents per episode
        n_random_agents = random.randint(1, 3)   # Random intruder agents
        
        logger.debug(f"Episode #{num_episodes}: Spawning {n_agents} planned agents + {n_random_agents} random agents")
        
        # Generate scenario (may be crossing, merging, diverging, or parser)
        
        routes_data = self._generate_mixed_scenario(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            n_agents=n_agents,
            n_random_agents=n_random_agents,
            json_path=parser_path
        )

        
        # Count actual agents in routes_data
        actual_n_agents = len(routes_data)
        
        # Create agents for this scenario
        self.all_agents = self._create_agents(actual_n_agents, 0)
        
        # Shuffle agents for random assignment
        all_agents_list = self.all_agents.get_all_agents()
        random.shuffle(all_agents_list)
        
        for i in range(len(all_agents_list)):
            agent = all_agents_list[i]
            route_info = routes_data[i]
            agent.waypoint_queue = route_info['waypoints']
            start_lat = route_info['start_lat']
            start_lon = route_info['start_lon']
            start_heading = route_info['start_heading']
            speed = route_info['speed']
            
            self.sim.traf_create(
                agent.id,
                actype="A320",
                acspd=speed,
                aclat=start_lat,
                aclon=start_lon,
                achdg=start_heading,
                acalt=FLIGHT_LEVEL
            )
            
            if agent != self.all_agents.get_active_agent():
                for waypoint in agent.waypoint_queue:
                    self.sim.traf_add_waypoint(agent.id, waypoint.lat, waypoint.lon, FLIGHT_LEVEL)
        
        self.sim.sim_step(float(0.1))
    
    def getNextWaypoint(self, agent: Agent) -> Waypoint:
        """Get next waypoint for agent"""
        if len(agent.waypoint_queue) == 0:
            return Waypoint(self.center_lat, self.center_lon)
        return agent.waypoint_queue[0]
    
    # ==================== RENDERING ====================
    
    def _render_frame(self):
        """Render the environment frame (shared across all environments)"""
        # Initialize canvas with RGBA support
        canvas = pygame.Surface(self.window_size, pygame.SRCALPHA)
        canvas.fill((169, 169, 169, 255))
        
        # Update camera position
        agent_positions = self.sim.traf_get_all()[:2]
        if len(agent_positions) > 0:
            self.camera.fixed_camera(CENTER_LAT, CENTER_LON)
        
        # Get active agent for later use
        active_agent = self.all_agents.get_active_agent()
        if active_agent:
            active_agent.update_observation_cache(self)
        
        # Render layers in order
        self._render_agent_trails(canvas)
        #self._render_routes(canvas, active_agent)
        self._render_waypoints(canvas, active_agent)
        self._render_intruders(canvas, active_agent)
        self._render_aircraft(canvas, active_agent)
        self._render_ui_elements(canvas, active_agent)
        
        # Display and handle events
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        self._handle_events()
    
    def _render_agent_trails(self, canvas):
        """Render agent movement trails with color-coded action types"""
        for agent in self.all_agents.get_all_agents():
            lat, lon, hdg, tas = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(self.sim, lat, lon, self.window_width, self.window_height)
            
            # Update trail data
            if agent.id not in self.agent_trails:
                self.agent_trails[agent.id] = []
            
            if x_pos is not None and y_pos is not None:
                action_type = getattr(agent, 'last_action_type', 'noop')
                self.agent_trails[agent.id].append((int(x_pos), int(y_pos), action_type))
                if len(self.agent_trails[agent.id]) > MAX_AGENT_TRAILS:
                    self.agent_trails[agent.id].pop(0)
            
            # Draw trail segments directly on canvas (opaque for performance)
            if len(self.agent_trails[agent.id]) > 1:
                trail_points = self.agent_trails[agent.id]
                for i in range(len(trail_points) - 1):
                    x1, y1, _ = trail_points[i]
                    x2, y2, action_type_end = trail_points[i + 1]
                    
                    if not (isinstance(x1, int) and isinstance(y1, int) and 
                           isinstance(x2, int) and isinstance(y2, int)):
                        continue
                    
                    # Color by action type: noop=red, snap=blue, steer=green
                    if action_type_end == 'noop':
                        seg_color = (255, 0, 0)
                    elif action_type_end == 'snap':
                        seg_color = (0, 120, 255)
                    else:  # steer
                        seg_color = (0, 180, 0)
                    
                    pygame.draw.line(canvas, seg_color, (x1, y1), (x2, y2), 2)
    
    def _render_routes(self, canvas, active_agent):
        """Render planned route lines"""
        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        
        for agent in self.all_agents.get_all_agents():
            if len(agent.waypoint_queue) == 0:
                continue
            
            lat, lon, hdg, tas = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(self.sim, lat, lon, self.window_width, self.window_height)
            
            # Build route points
            route_points = []
            if x_pos is not None and y_pos is not None:
                try:
                    route_points.append((x_pos, y_pos))
                except (ValueError, TypeError):
                    pass
            
            for waypoint in agent.waypoint_queue:
                target_x, target_y = self.camera.latlon_to_screen(
                    self.sim, waypoint.lat, waypoint.lon, self.window_width, self.window_height
                )
                if target_x is not None and target_y is not None:
                    try:
                        route_points.append((target_x, target_y))
                    except (ValueError, TypeError):
                        pass
            
            # Draw route lines
            if len(route_points) >= 2:
                try:
                    for i in range(len(route_points) - 1):
                        x1, y1 = int(route_points[i][0]), int(route_points[i][1])
                        x2, y2 = int(route_points[i+1][0]), int(route_points[i+1][1])
                        pygame.draw.line(overlay, (0, 255, 0, 65), (x1, y1), (x2, y2), 1)
                except Exception:
                    pass
            elif len(route_points) == 1:
                try:
                    pygame.draw.circle(overlay, (0, 255, 0, 180), 
                                     (int(route_points[0][0]), int(route_points[0][1])), 5)
                except Exception:
                    pass
        
        canvas.blit(overlay, (0, 0))
    
    def _render_waypoints(self, canvas, active_agent):
        """Render waypoint markers for active agent"""
        if not active_agent or len(active_agent.waypoint_queue) == 0:
            return
        
        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        
        for waypoint_idx, waypoint in enumerate(active_agent.waypoint_queue):
            target_x, target_y = self.camera.latlon_to_screen(
                self.sim, waypoint.lat, waypoint.lon, self.window_width, self.window_height
            )
            
            if target_x is None or target_y is None:
                continue
            if not (0 <= int(target_x) <= self.window_width and 
                   0 <= int(target_y) <= self.window_height):
                continue
            
            # Next waypoint = white, others = dark grey
            if waypoint_idx == 0:
                color = (255, 255, 255, 255)
                radius = 6
            else:
                color = (100, 100, 100, 180)
                radius = 4
            
            pygame.draw.circle(overlay, color, (int(target_x), int(target_y)), 
                             radius=radius, width=0)
            
            # Draw margin circle for next waypoint
            if waypoint_idx == 0:
                target_margin_radius = (TARGET_DISTANCE_MARGIN * NM2KM / 
                                      self.camera.zoom_km) * self.window_width
                if target_margin_radius > 2:
                    pygame.draw.circle(overlay, color, (int(target_x), int(target_y)), 
                                     radius=int(target_margin_radius), width=2)
        
        canvas.blit(overlay, (0, 0))
    
    def _render_intruders(self, canvas, active_agent):
        """Render intruder aircraft markers and collision information"""
        if not active_agent:
            return
        
        # Collect all intruders
        all_intruder_ids = []
        if hasattr(active_agent, 'intruder_collision_cache'):
            all_intruder_ids = list(active_agent.intruder_collision_cache.keys())
        selected_set = set(getattr(active_agent, 'selected_intruder_ids', []))
        
        # Render selected intruders (detailed info)
        if hasattr(active_agent, 'selected_intruder_ids') and active_agent.selected_intruder_ids:
            self._render_selected_intruders(canvas, active_agent)
        
        # Render non-selected intruders (simple markers)
        self._render_other_intruders(canvas, active_agent, all_intruder_ids, selected_set)
    
    def _render_selected_intruders(self, canvas, active_agent):
        """Render detailed information for selected intruders"""
        font_obs = pygame.font.SysFont(None, 14)
        cpa_font = pygame.font.SysFont(None, 16)
        
        for slot_idx, intruder_id in enumerate(active_agent.selected_intruder_ids):
            if not hasattr(active_agent, 'intruder_state_map'):
                continue
            if intruder_id not in active_agent.intruder_state_map:
                continue
            
            int_lat, int_lon, int_hdg, int_tas = active_agent.intruder_state_map[intruder_id]
            int_x, int_y = self.camera.latlon_to_screen(
                self.sim, int_lat, int_lon, self.window_width, self.window_height
            )
            
            if int_x is None or int_y is None:
                continue
            if not (0 <= int_x <= self.window_width and 0 <= int_y <= self.window_height):
                continue
            
            # Determine threat level and color
            obs_color = (255, 165, 0, 255)  # Default: Orange
            marker_size = 12
            ring_width = 3
            
            if hasattr(active_agent, 'intruder_collision_cache') and \
               intruder_id in active_agent.intruder_collision_cache:
                collision_info = active_agent.intruder_collision_cache[intruder_id]
                time_to_min_sep = collision_info.get('time_to_min_sep', 0)
                closing_rate = collision_info.get('closing_rate', 0)
                min_separation = collision_info.get('min_separation', 100)
                
                is_real_threat = self._is_actual_danger(closing_rate, time_to_min_sep, min_separation)
                if is_real_threat:
                    obs_color = (255, 0, 0, 255)  # Red for real conflicts
                    marker_size = 14
                    ring_width = 3
            else:
                collision_info = {}
            
            # Draw marker circle
            pygame.draw.circle(canvas, obs_color, (int(int_x), int(int_y)), marker_size, ring_width)
            
            # Draw slot index
            slot_text = f"[{slot_idx}]"
            text_surface = font_obs.render(slot_text, True, obs_color[:3])
            canvas.blit(text_surface, (int(int_x) + marker_size + 3, int(int_y) - 8))
            
            # Draw intruder ID
            id_text = intruder_id[-3:]
            text_surface = font_obs.render(id_text, True, obs_color[:3])
            canvas.blit(text_surface, (int(int_x) + marker_size + 3, int(int_y) + 5))
            
            # Draw collision info
            if collision_info:
                time_to_min_sep = collision_info.get('time_to_min_sep', 0)
                closing_rate = collision_info.get('closing_rate', 0)
                min_separation = collision_info.get('min_separation', 100)
                
                logger.debug(f"[RENDER] Slot {slot_idx} ({intruder_id}): "
                           f"min_sep={min_separation:.2f}NM, time={time_to_min_sep:.1f}s, "
                           f"cr={closing_rate:+.2f}m/s")
                
                # Time to CPA
                if time_to_min_sep < 0:
                    time_text = f"t:PAST {time_to_min_sep:.0f}s"
                elif time_to_min_sep >= 900:
                    time_text = f"t:900s+"
                else:
                    time_text = f"t:{time_to_min_sep:.1f}s"
                time_surface = cpa_font.render(time_text, True, obs_color[:3])
                canvas.blit(time_surface, (int(int_x) + marker_size + 3, int(int_y) + 15))
                
                # Closing rate
                cr_text = f"cr:{closing_rate:+.2f}m/s"
                cr_surface = cpa_font.render(cr_text, True, obs_color[:3])
                canvas.blit(cr_surface, (int(int_x) + marker_size + 3, int(int_y) + 30))
                
                # Minimum separation
                min_sep_text = f"sep:{min_separation:.2f}NM"
                min_sep_surface = cpa_font.render(min_sep_text, True, obs_color[:3])
                canvas.blit(min_sep_surface, (int(int_x) + marker_size + 3, int(int_y) + 45))
                
                # Draw CPA position marker
                self._render_cpa_marker(canvas, active_agent, intruder_id, 
                                       time_to_min_sep, obs_color)
    
    def _render_cpa_marker(self, canvas, active_agent, intruder_id, time_to_min_sep, color):
        """Render CPA (Closest Point of Approach) position marker"""
        if time_to_min_sep <= 0 or time_to_min_sep >= LONG_CONFLICT_THRESHOLD_SEC:
            return
        
        if not hasattr(active_agent, 'cpa_position_map') or \
           intruder_id not in active_agent.cpa_position_map:
            return
        
        cpa_lat, cpa_lon = active_agent.cpa_position_map[intruder_id]
        if cpa_lat is None or cpa_lon is None:
            return
        
        active_lat, active_lon, _, _ = self.sim.traf_get_state(active_agent.id)
        active_x, active_y = self.camera.latlon_to_screen(
            self.sim, active_lat, active_lon, self.window_width, self.window_height
        )
        cpa_x, cpa_y = self.camera.latlon_to_screen(
            self.sim, cpa_lat, cpa_lon, self.window_width, self.window_height
        )
        
        if cpa_x is None or cpa_y is None or active_x is None or active_y is None:
            return
        if not (0 <= cpa_x <= self.window_width and 0 <= cpa_y <= self.window_height):
            return
        
        # Draw CPA marker and connecting line directly
        cpa_point_color = (color[0], color[1], color[2])
        pygame.draw.circle(canvas, cpa_point_color, (int(cpa_x), int(cpa_y)), 6, 2)
        pygame.draw.line(canvas, cpa_point_color, 
                        (int(active_x), int(active_y)), (int(cpa_x), int(cpa_y)), 1)
    
    def _render_other_intruders(self, canvas, active_agent, all_intruder_ids, selected_set):
        """Render simple markers for non-selected intruders"""
        font_other = pygame.font.SysFont(None, 12)
        
        for intruder_id in all_intruder_ids:
            if intruder_id in selected_set:
                continue
            if not hasattr(active_agent, 'intruder_state_map'):
                continue
            if intruder_id not in active_agent.intruder_state_map:
                continue
            
            int_lat, int_lon, int_hdg, int_tas = active_agent.intruder_state_map[intruder_id]
            int_x, int_y = self.camera.latlon_to_screen(
                self.sim, int_lat, int_lon, self.window_width, self.window_height
            )
            
            if int_x is None or int_y is None:
                continue
            if not (0 <= int_x <= self.window_width and 0 <= int_y <= self.window_height):
                continue
            
            # Determine color based on threat level
            base_color = (140, 140, 140)
            marker_size = 9
            ring_width = 2
            
            collision_info = active_agent.intruder_collision_cache.get(intruder_id, {})
            closing_rate = collision_info.get('closing_rate', 0)
            time_to_min_sep = collision_info.get('time_to_min_sep', 0)
            min_separation = collision_info.get('min_separation', 999)
            
            is_real_threat = self._is_actual_danger(closing_rate, time_to_min_sep, min_separation)
            if is_real_threat:
                base_color = (255, 0, 0)
                marker_size = 11
            elif collision_info.get('future_collision', False):
                base_color = (255, 165, 0)
            
            # Draw marker directly
            pygame.draw.circle(canvas, base_color, (int(int_x), int(int_y)), marker_size, ring_width)
            
            # Draw ID text
            id_text = intruder_id[-3:]
            txt = font_other.render(id_text, True, base_color)
            canvas.blit(txt, (int(int_x) + marker_size + 2, int(int_y) - 6))
    
    def _render_aircraft(self, canvas, active_agent):
        """Render aircraft symbols, headings, and CPA fan visualization"""
        for agent in self.all_agents.get_all_agents():
            lat, lon, hdg, tas = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(
                self.sim, lat, lon, self.window_width, self.window_height
            )
            
            if x_pos is None or y_pos is None:
                continue
            if not (0 <= x_pos <= self.window_width and 0 <= y_pos <= self.window_height):
                continue
            
            # Determine color: Yellow for active agent, grey for others
            is_active = agent == active_agent
            color = (255, 255, 0, 255) if is_active else (80, 80, 80, 200)
            
            # Draw heading line
            line_length_km = 12 / NM2KM
            lat_end, lon_end = fn.get_point_at_distance(lat, lon, line_length_km, hdg)
            heading_x, heading_y = self.camera.latlon_to_screen(
                self.sim, lat_end, lon_end, self.window_width, self.window_height
            )
            pygame.draw.line(canvas, (0, 0, 0, 255), (x_pos, y_pos), 
                           (heading_x, heading_y), 2)
            
            # Draw multi-heading CPA fan for active agent
            if is_active and MULTI_CAP_HEADING_RENDER:
                self._render_cpa_fan(canvas, agent, x_pos, y_pos, lat, lon, hdg)
            
            # Draw intrusion distance circle
            dist_km = INTRUSION_DISTANCE * NM2KM / 2
            radius_px = (dist_km / self.camera.zoom_km) * self.window_width
            if radius_px > 1:
                pygame.draw.circle(canvas, color[:3], (int(x_pos), int(y_pos)), 
                                 int(radius_px), 2)
            
            # Draw speed text
            font = pygame.font.SysFont(None, 20)
            tas_text = f"{tas:.0f}m/s"
            text_surface = font.render(tas_text, True, (0, 0, 0))
            canvas.blit(text_surface, (int(x_pos + radius_px + 5), int(y_pos - 10)))
    
    def _render_cpa_fan(self, canvas, agent, x_pos, y_pos, lat, lon, hdg):
        """Render multi-heading CPA visualization fan"""
        obs_dict = agent.obs_dict_cache if agent.obs_dict_cache is not None else \
                   self._get_observation_dict(agent)
        multi_heading_cpa = obs_dict.get('multi_heading_cpa', None)
        
        if multi_heading_cpa is None or len(multi_heading_cpa) != NUM_HEADING_OFFSETS:
            return
        
        fan_radius_nm = 30.0
        fan_radius_km = fan_radius_nm * NM2KM
        fan_radius_px = (fan_radius_km / self.camera.zoom_km) * self.window_width
        segment_half_angle = 5.0
        
        fan_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        
        for offset_idx in range(NUM_HEADING_OFFSETS):
            offset_deg = HEADING_OFFSETS[offset_idx]
            cpa_value = multi_heading_cpa[offset_idx]
            
            # Color gradient: Green (safe) -> Yellow -> Red (dangerous)
            if cpa_value >= 0.8:
                r = int(255 * (1.0 - cpa_value) / 0.2)
                g = 255
                b = 0
            elif cpa_value >= 0.5:
                r = 255
                g = int(255 * (cpa_value - 0.5) / 0.3)
                b = 0
            else:
                r = 255
                g = 0
                b = 0
            
            segment_color = (r, g, b, 120)
            
            # Calculate segment boundaries
            heading_with_offset = hdg + offset_deg
            angle_left = heading_with_offset - segment_half_angle
            angle_right = heading_with_offset + segment_half_angle
            
            lat_left, lon_left = fn.get_point_at_distance(lat, lon, fan_radius_km, angle_left)
            lat_right, lon_right = fn.get_point_at_distance(lat, lon, fan_radius_km, angle_right)
            
            x_left, y_left = self.camera.latlon_to_screen(
                self.sim, lat_left, lon_left, self.window_width, self.window_height
            )
            x_right, y_right = self.camera.latlon_to_screen(
                self.sim, lat_right, lon_right, self.window_width, self.window_height
            )
            
            if x_left is None or y_left is None or x_right is None or y_right is None:
                continue
            
            # Draw fan segment
            pygame.draw.polygon(fan_surface, segment_color, 
                              [(int(x_pos), int(y_pos)), 
                               (int(x_left), int(y_left)), 
                               (int(x_right), int(y_right))], 0)
            
            # Draw segment borders
            pygame.draw.line(fan_surface, (0, 0, 0, 80), 
                           (int(x_pos), int(y_pos)), (int(x_left), int(y_left)), 1)
            pygame.draw.line(fan_surface, (0, 0, 0, 80), 
                           (int(x_pos), int(y_pos)), (int(x_right), int(y_right)), 1)
            pygame.draw.line(fan_surface, (0, 0, 0, 80), 
                           (int(x_left), int(y_left)), (int(x_right), int(y_right)), 1)
        
        canvas.blit(fan_surface, (0, 0))
    
    def _render_ui_elements(self, canvas, active_agent):
        """Render UI overlays: time, scale bar, debug info, pause overlay"""
        # Simulation time
        sim_time = self.steps * AGENT_INTERACTION_TIME
        start_hour = 12
        current_hour = int(start_hour + sim_time // 3600)
        current_minute = int((sim_time % 3600) // 60)
        current_second = int(sim_time % 60)
        time_str = f"{current_hour:02d}:{current_minute:02d}:{current_second:02d}"
        
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(time_str, True, (0, 0, 0))
        text_rect = text_surface.get_rect()
        text_rect.bottomright = (self.window_width - 20, self.window_height - 20)
        canvas.blit(text_surface, text_rect)
        
        # Scale bar
        self._render_scale_bar(canvas)
        
        # Debug information
        self._render_debug_info(canvas, active_agent)
        
        # Pause overlay
        if self.is_paused:
            self._render_pause_overlay(canvas)
    
    def _render_scale_bar(self, canvas):
        """Render 1 NM scale bar"""
        one_nm_km = 1.0
        one_nm_px = (one_nm_km / self.camera.zoom_km) * self.window_width
        
        legend_bottom = self.window_height - 50
        legend_left = 20
        
        # Horizontal line
        pygame.draw.line(canvas, (0, 0, 0, 255), 
                        (legend_left, legend_bottom), 
                        (legend_left + one_nm_px, legend_bottom), 
                        width=3)
        
        # End markers
        pygame.draw.line(canvas, (0, 0, 0, 255), 
                        (legend_left, legend_bottom - 5), 
                        (legend_left, legend_bottom + 5), 
                        width=2)
        pygame.draw.line(canvas, (0, 0, 0, 255), 
                        (legend_left + one_nm_px, legend_bottom - 5), 
                        (legend_left + one_nm_px, legend_bottom + 5), 
                        width=2)
        
        # Label
        legend_font_small = pygame.font.SysFont(None, 16)
        text_surface = legend_font_small.render("1 NM", True, (0, 0, 0))
        text_x = legend_left + (one_nm_px / 2) - (text_surface.get_width() / 2)
        text_y = legend_bottom + 10
        canvas.blit(text_surface, (text_x, text_y))
    
    def _render_pause_overlay(self, canvas):
        """Render pause screen overlay"""
        # Semi-transparent dark overlay (this one overlay is acceptable for rare pause state)
        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        canvas.blit(overlay, (0, 0))
        
        font_pause = pygame.font.SysFont(None, 48, bold=True)
        pause_text = font_pause.render("PAUSED - Press SPACE to resume", True, (255, 0, 0))
        text_rect = pause_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
        
        # Background box
        pygame.draw.rect(canvas, (255, 255, 255), text_rect.inflate(20, 20), border_radius=10)
        canvas.blit(pause_text, text_rect)
    
    def _handle_events(self):
        """Handle pygame events (quit, pause)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
        
        # Pause loop
        while self.is_paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.is_paused = False
            pygame.time.wait(100)
    
    def _render_debug_info(self, canvas, active_agent):
        """Render debug information (observation space and rewards) on canvas"""
        if not active_agent:
            return
        
        obs_dict = active_agent.obs_dict_cache if active_agent.obs_dict_cache is not None else self._get_observation_dict(active_agent)
        reward_comp = active_agent.last_reward_components if hasattr(active_agent, 'last_reward_components') and active_agent.last_reward_components else {}
        info = self._get_info()
        
        font_title = pygame.font.SysFont(None, 20, bold=True)
        font_normal = pygame.font.SysFont(None, 16)
        font_small = pygame.font.SysFont(None, 14)
        
        debug_y = self.window_height - 250
        debug_x_left = 10
        debug_x_right = self.window_width // 2 + 10
        
        # Left side: Observation Space
        title_text = f"OBS SPACE - {active_agent.id} (Step {self.steps})"
        text_surf = font_title.render(title_text, True, (0, 0, 0))
        canvas.blit(text_surf, (debug_x_left, debug_y))
        debug_y += 25
        
        # Ego State
        ego_state = obs_dict['ego_state']
        text = f"EGO: Hdg({ego_state[0]:+.2f},{ego_state[1]:+.2f}) Spd({ego_state[2]:+.2f}) Drift({ego_state[3]:+.2f}) DstWp({ego_state[4]:+.2f})"
        text_surf = font_small.render(text, True, (0, 0, 0))
        canvas.blit(text_surf, (debug_x_left, debug_y))
        debug_y += 18
        
        # Threat Features
        threat_features = obs_dict['threat_features']
        text = f"THREATS ({NUM_AC_STATE} slots):"
        text_surf = font_normal.render(text, True, (255, 0, 0))
        canvas.blit(text_surf, (debug_x_left, debug_y))
        debug_y += 18
        
        for slot_idx in range(NUM_AC_STATE):
            slot_start = slot_idx * 9
            slot_data = threat_features[slot_start:slot_start+9]
            
            intruder_id = active_agent.selected_intruder_ids[slot_idx] if slot_idx < len(active_agent.selected_intruder_ids) else "FILLER"
            is_filler = np.allclose(slot_data[:6], [1.0, -1.0, 0.0, 0.0, 1.0, 1.0], atol=0.05)
            slot_type = "FILL" if is_filler else "THREAT"
            color = (100, 100, 100) if is_filler else (255, 100, 0)
            
            text = f"  [{slot_idx}]{intruder_id}({slot_type}): Dst({slot_data[0]:+.2f}) Brg({slot_data[1]:+.2f},{slot_data[2]:+.2f}) CR({slot_data[3]:+.2f}) MinSep({slot_data[4]:+.2f}) T({slot_data[5]:+.2f}) TAS({slot_data[6]:+.0f}) Rel({slot_data[7]:+.3f},{slot_data[8]:+.3f})"
            text_surf = font_small.render(text, True, color)
            canvas.blit(text_surf, (debug_x_left, debug_y))
            debug_y += 16
        
        # Right side: Rewards
        debug_y_right = 10
        
        title_text = "REWARDS"
        text_surf = font_title.render(title_text, True, (0, 0, 0))
        canvas.blit(text_surf, (debug_x_right, debug_y_right))
        debug_y_right += 25
        
        drift_r = reward_comp.get('drift', 0.0)
        cpa_warning_r = reward_comp.get('cpa_warning', 0.0)
        proximity_r = reward_comp.get('proximity', 0.0)
        total_r = reward_comp.get('total', 0.0)
        
        text = f"Drift:        {drift_r:+.4f}"
        color = (0, 200, 0) if drift_r > 0 else (200, 0, 0)
        text_surf = font_normal.render(text, True, color)
        canvas.blit(text_surf, (debug_x_right, debug_y_right))
        debug_y_right += 20
        
        text = f"CPA Warning:  {cpa_warning_r:+.4f}"
        color = (0, 200, 0) if cpa_warning_r > 0 else (200, 0, 0)
        text_surf = font_normal.render(text, True, color)
        canvas.blit(text_surf, (debug_x_right, debug_y_right))
        debug_y_right += 20
        
        text = f"Proximity:    {proximity_r:+.4f}"
        color = (0, 200, 0) if proximity_r > 0 else (200, 0, 0)
        text_surf = font_normal.render(text, True, color)
        canvas.blit(text_surf, (debug_x_right, debug_y_right))
        debug_y_right += 20
        
        text = f"TOTAL: {total_r:+.4f}"
        color = (0, 255, 0) if total_r > 0 else (255, 0, 0)
        text_surf = font_title.render(text, True, color)
        canvas.blit(text_surf, (debug_x_right, debug_y_right))
        debug_y_right += 28
        
        debug_y_right += 5
        text = f"Ep: {info['num_episodes']} | Intrusions: {info['intrusions']}"
        text_surf = font_small.render(text, True, (0, 0, 0))
        canvas.blit(text_surf, (debug_x_right, debug_y_right))
        debug_y_right += 18
        
        text = f"Waypoints: {info['waypoints_collected']} | Avg Reward: {info['avg_reward']:+.3f}"
        text_surf = font_small.render(text, True, (0, 0, 0))
        canvas.blit(text_surf, (debug_x_right, debug_y_right))

    def render(self, mode="human"):
        """Render the environment"""
        self.render_mode = mode
        if mode == "human":
            self._render_frame()
            time.sleep(0.05)
        elif mode == "rgb_array":
            logger.error("RGB array rendering not implemented")
            return np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
    
    def close(self):
        """Close pygame window"""
        if self.window is not None:
            pygame.quit()
            self.window = None

    def _set_action(self, action, agent: Agent) -> None:
        """Set the action for the agent (heading change)"""
        agent.action_age += 1
    