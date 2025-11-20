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
from typing import Dict, List, Optional
from collections import deque

import bluesky_gym.envs.common.functions as fn

from .helper_classes import Waypoint, Agent, Agents, Camera
from .consts import *
from .helper_functions import (
    compute_cpa_multi_heading_numba,
    bound_angle_positive_negative_180,
    get_point_at_distance
)
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
        """Vectorized collision criticality calculation"""
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
            INTRUSION_DISTANCE,
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
    
    # ==================== MULTI-HEADING CPA ====================
    
    def _compute_multi_heading_cpa_features(
        self, agent: Agent, ac_lat: float, ac_lon: float, ac_hdg: float, ac_tas: float
    ) -> np.ndarray:
        """Compute multi-heading CPA features (vectorized)"""
        multi_heading_cpa_features = np.ones(NUM_HEADING_OFFSETS, dtype=np.float32)
        
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
        
        min_seps, t_cpas, c_rates = compute_cpa_multi_heading_numba(
            ac_hdg, ac_tas,
            intruder_hdgs, intruder_tas_array,
            bearings, distances,
            INTRUSION_DISTANCE,
            HEADING_OFFSETS
        )
        
        is_critical = self._is_actual_danger(c_rates, t_cpas, min_seps)
        t_cpas_masked = np.where(is_critical, t_cpas, np.inf)
        min_critical_times = np.min(t_cpas_masked, axis=0)
        
        valid_mask = min_critical_times < np.inf
        multi_heading_cpa_features[valid_mask] = np.clip(
            min_critical_times[valid_mask] / LONG_CONFLICT_THRESHOLD_SEC, 0.0, 1.0
        )
        
        del bearings, distances, min_seps, t_cpas, c_rates, is_critical, t_cpas_masked
        return multi_heading_cpa_features
    
    # ==================== INTRUDER SELECTION & FEATURES ====================
    
    def _compute_action_history(self, agent: Agent) -> np.ndarray:
        """
        Compute action history features (5 features):
        - last_action_do: [0, 1] - War letzte Aktion ein Lenken?
        - last_action_continuous: [-1, 1] - Lenkintensität
        - action_age_normalized: [0, 1] - Alter der Aktion
        - turning_rate_normalized: [-1, 1] - Aktuelle Rotationsrate
        - last_action_speed: [-1, 0, 1] - Normalisiert (0=Beschleunigen, 1=Nichts, 2=Verlangsamen)
        """
        last_action_do = float(agent.last_action)
        last_action_continuous = float(agent.last_action_continuous)
        action_age_normalized = float(agent.action_age)
        turning_rate_normalized = float(agent.turning_rate)
        
        # Speed action normalisiert: 0→-1 (Beschleunigen), 1→0 (Nichts), 2→+1 (Verlangsamen)
        speed_action_normalized = float(agent.last_action_speed) - 1.0
        
        return np.array([
            last_action_do, 
            last_action_continuous, 
            action_age_normalized,
            turning_rate_normalized,
            speed_action_normalized
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
        
        ego_state = np.array([
            ego_hdg_cos, ego_hdg_sin,
            speed_normalized,
            drift_normalized,
            vertical_separation_normalized,
            distance_to_waypoint_normalized
        ], dtype=np.float32)
        
        # Threat Features
        threat_features = self._compute_intruder_features(agent, ac_lat, ac_lon, ac_hdg, ac_tas)
        
        # Multi-Heading CPA
        multi_heading_cpa = self._compute_multi_heading_cpa_features(agent, ac_lat, ac_lon, ac_hdg, ac_tas)
        
        # Action History
        action_history = self._compute_action_history(agent)
        
        return {
            'ego_state': ego_state,
            'threat_features': threat_features,
            'action_history': action_history,
            'multi_heading_cpa': multi_heading_cpa
        }
    
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
    
    # ==================== AIRCRAFT GENERATION ====================
    
    def _create_agents(self, n_agents, n_random_agents):
        """Create agent list"""
        agents = Agents()
        total_agents = n_agents + n_random_agents
        for i in range(total_agents):
            agents.add_agent(Agent(f'kl00{i+1}'.upper(), 0.0, False, 999.0, 0.0))
        return agents
    
    def _generate_crossing_routes(self, center_lat, center_lon, n_agents, n_random_agents):
        """Generate crossing routes with ring distribution"""
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
                'speed': common_speed,
                'ring_idx': assigned_ring
            }
        
        # Random agents
        outer_ring_radius_nm = 145
        outer_ring_radius_km = outer_ring_radius_nm * NM2KM
        
        for random_agent_idx in range(n_random_agents):
            agent_idx = n_agents + random_agent_idx
            
            offset_x_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
            offset_y_km = random.uniform(-outer_ring_radius_km, outer_ring_radius_km)
            start_lat = center_lat + (offset_x_km / 110.54)
            start_lon = center_lon + (offset_y_km / 111.32)
            start_heading = random.uniform(0, 360)
            
            waypoints = deque()
            
            exit_heading = random.uniform(0, 360)
            exit_distance_nm = random.uniform(50, 200)
            exit_distance_km = exit_distance_nm * NM2KM
            exit_angle_rad = np.deg2rad(exit_heading)
            exit_x_km = exit_distance_km * np.cos(exit_angle_rad)
            exit_y_km = exit_distance_km * np.sin(exit_angle_rad)
            exit_lat = start_lat + (exit_x_km / 110.54)
            exit_lon = start_lon + (exit_y_km / 111.32)
            waypoints.append(Waypoint(exit_lat, exit_lon))
            
            routes_dict[agent_idx] = {
                'waypoints': waypoints,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'start_heading': start_heading,
                'speed': common_speed,
                'ring_idx': -1
            }
        
        return routes_dict
    
    def _gen_aircraft(self, num_episodes: int):
        """Generate aircraft with randomized distribution"""
        n_agents = random.randint(3, 7)
        n_random_agents = random.randint(0, 3)
        
        logger.debug(f"Episode #{num_episodes}: Spawning {n_agents} ring agents + {n_random_agents} random agents")
        
        self.all_agents = self._create_agents(n_agents, n_random_agents)
        
        routes_data = self._generate_crossing_routes(
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            n_agents=n_agents,
            n_random_agents=n_random_agents
        )
        
        spawn_positions = []
        
        for agent_idx, agent in enumerate(self.all_agents.get_all_agents()):
            route_info = routes_data[agent_idx]
            agent.waypoint_queue = route_info['waypoints']
            
            start_lat = route_info['start_lat']
            start_lon = route_info['start_lon']
            start_heading = route_info['start_heading']
            speed = route_info['speed']
            
            for other_lat, other_lon in spawn_positions:
                _, dist_nm = self.sim.geo_calculate_direction_and_distance(
                    start_lat, start_lon, other_lat, other_lon
                )
                if dist_nm < SPAWN_SEPARATION_MIN:
                    offset_dist = random.uniform(0.5, 2.0)
                    offset_angle = random.uniform(0, 360)
                    start_lat, start_lon = get_point_at_distance(
                        start_lat, start_lon, offset_dist, offset_angle
                    )
                    break
            
            spawn_positions.append((start_lat, start_lon))
            
            self.sim.traf_create(
                agent.id,
                actype="A320",
                acspd=np.clip(speed + random.uniform(-5, 5), MIN_SPEED, MAX_SPEED),
                aclat=start_lat,
                aclon=start_lon,
                achdg=start_heading,
                acalt=FLIGHT_LEVEL
            )
            
            if agent != self.all_agents.get_active_agent():
                for waypoint in agent.waypoint_queue:
                    self.sim.traf_add_waypoint(agent.id, waypoint.lat, waypoint.lon, FLIGHT_LEVEL)
                self.sim.traf_activate_lnav(agent.id)
        
        self.sim.sim_step(float(0.1))
    
    def getNextWaypoint(self, agent: Agent) -> Waypoint:
        """Get next waypoint for agent"""
        if len(agent.waypoint_queue) == 0:
            return Waypoint(self.center_lat, self.center_lon)
        return agent.waypoint_queue[0]
    
    # ==================== RENDERING ====================
    
    def _render_frame(self):
        """Render the environment frame (shared across all environments)"""
        canvas = pygame.Surface(self.window_size)
        canvas.fill((169, 169, 169)) 

        agent_positions = self.sim.traf_get_all()[:2]
        if len(agent_positions) > 0:
            self.camera.fixed_camera(CENTER_LAT, CENTER_LON)

        # Draw agent trails
        for agent in self.all_agents.get_all_agents():
            lat, lon, hdg, tas = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(self.sim, lat, lon, self.window_width, self.window_height)
            
            if agent.id not in self.agent_trails:
                self.agent_trails[agent.id] = []

            if x_pos is not None and y_pos is not None:
                self.agent_trails[agent.id].append((int(x_pos), int(y_pos)))
                if len(self.agent_trails[agent.id]) > MAX_AGENT_TRAILS:
                    self.agent_trails[agent.id].pop(0)

            if len(self.agent_trails[agent.id]) > 1:
                valid_trail = [(int(x), int(y)) for x, y in self.agent_trails[agent.id] 
                              if isinstance(x, int) and isinstance(y, int)]
                if len(valid_trail) > 1:
                    pygame.draw.lines(canvas, (0, 120, 255), False, valid_trail, 1)

        # Draw routes and waypoints
        for agent in self.all_agents.get_all_agents():
            lat, lon, hdg, tas = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(self.sim, lat, lon, self.window_width, self.window_height)
            
            if len(agent.waypoint_queue) > 0:
                waypoint_list = list(agent.waypoint_queue)
                route_points = []
                
                if x_pos is not None and y_pos is not None:
                    try:
                        route_points.append((x_pos, y_pos))
                    except (ValueError, TypeError):
                        pass
                
                for waypoint in waypoint_list:
                    target_x, target_y = self.camera.latlon_to_screen(self.sim, waypoint.lat, waypoint.lon, self.window_width, self.window_height)
                    if target_x is not None and target_y is not None:
                        try:
                            route_points.append((target_x, target_y))
                        except (ValueError, TypeError):
                            pass
                
                if len(route_points) >= 2:
                    try:
                        for i in range(len(route_points) - 1):
                            x1, y1 = int(route_points[i][0]), int(route_points[i][1])
                            x2, y2 = int(route_points[i+1][0]), int(route_points[i+1][1])
                            pygame.draw.line(canvas, (0, 255, 0), (x1, y1), (x2, y2), 1)
                    except Exception:
                        pass
                elif len(route_points) == 1:
                    try:
                        pygame.draw.circle(canvas, (0, 255, 0), (int(route_points[0][0]), int(route_points[0][1])), 5)
                    except Exception:
                        pass
            
            if agent == self.all_agents.get_active_agent():
                for waypoint_idx, waypoint in enumerate(agent.waypoint_queue):
                    target_x, target_y = self.camera.latlon_to_screen(self.sim, waypoint.lat, waypoint.lon, self.window_width, self.window_height)
                    
                    if target_x is None or target_y is None:
                        continue
                    
                    if waypoint_idx == 0:
                        color = (255, 255, 255)  # Weiß für nächsten Wegpunkt
                        radius = 6
                    else:
                        color = (100, 100, 100)  # Dunkelgrau für weitere
                        radius = 4
                    
                    if 0 <= int(target_x) <= self.window_width and 0 <= int(target_y) <= self.window_height:
                        pygame.draw.circle(canvas, color, (int(target_x), int(target_y)), radius=radius, width=0)
                        target_margin_radius = (TARGET_DISTANCE_MARGIN * NM2KM / self.camera.zoom_km) * self.window_width
                        if target_margin_radius > 2 and waypoint_idx == 0:
                            pygame.draw.circle(canvas, color, (int(target_x), int(target_y)), radius=int(target_margin_radius), width=2)

        active_agent = self.all_agents.get_active_agent()

        if active_agent:
            active_agent.update_observation_cache(self)
        
        # Draw intruders/threats
        if active_agent and hasattr(active_agent, 'selected_intruder_ids') and active_agent.selected_intruder_ids:
            font_obs = pygame.font.SysFont(None, 14)
            
            for slot_idx, intruder_id in enumerate(active_agent.selected_intruder_ids):
                if not hasattr(active_agent, 'intruder_state_map'):
                    continue
                if intruder_id not in active_agent.intruder_state_map:
                    continue
                
                int_lat, int_lon, int_hdg, int_tas = active_agent.intruder_state_map[intruder_id]
                int_x, int_y = self.camera.latlon_to_screen(self.sim, int_lat, int_lon, self.window_width, self.window_height)
                
                if int_x is None or int_y is None:
                    continue
                
                if not (0 <= int_x <= self.window_width and 0 <= int_y <= self.window_height):
                    continue
                
                obs_color = (255, 165, 0)  # Default: Orange
                marker_size = 12
                ring_width = 3
                
                if hasattr(active_agent, 'intruder_collision_cache') and intruder_id in active_agent.intruder_collision_cache:
                    collision_info = active_agent.intruder_collision_cache[intruder_id]
                    time_to_min_sep = collision_info.get('time_to_min_sep', 0)
                    closing_rate = collision_info.get('closing_rate', 0)
                    min_separation = collision_info.get('min_separation', 100)

                    is_real_threat = self._is_actual_danger(closing_rate, time_to_min_sep, min_separation)
                    if is_real_threat:
                        obs_color = (255, 0, 0)  # ROT für echte Konflikte
                        marker_size = 14
                        ring_width = 3
                
                pygame.draw.circle(canvas, obs_color, (int(int_x), int(int_y)), marker_size, ring_width)
                
                slot_text = f"[{slot_idx}]"
                text_surface = font_obs.render(slot_text, True, obs_color)
                text_x = int_x + marker_size + 3
                text_y = int_y - 8
                canvas.blit(text_surface, (int(text_x), int(text_y)))
                
                id_text = intruder_id[-3:]
                text_surface = font_obs.render(id_text, True, obs_color)
                text_x = int_x + marker_size + 3
                text_y = int_y + 5
                canvas.blit(text_surface, (int(text_x), int(text_y)))
                
                if hasattr(active_agent, 'intruder_collision_cache') and intruder_id in active_agent.intruder_collision_cache:
                    collision_info = active_agent.intruder_collision_cache[intruder_id]
                    time_to_min_sep = collision_info.get('time_to_min_sep', 0)
                    closing_rate = collision_info.get('closing_rate', 0)
                    min_separation = collision_info.get('min_separation', 100)
                    
                    logger.debug(f"[RENDER] Slot {slot_idx} ({intruder_id}): min_sep={min_separation:.2f}NM, time={time_to_min_sep:.1f}s, cr={closing_rate:+.2f}m/s")
                    
                    cpa_font = pygame.font.SysFont(None, 16)
                    
                    if time_to_min_sep < 0:
                        time_text = f"t:PAST {time_to_min_sep:.0f}s"
                    elif time_to_min_sep >= 900:
                        time_text = f"t:900s+"
                    else:
                        time_text = f"t:{time_to_min_sep:.1f}s"
                    time_surface = cpa_font.render(time_text, True, obs_color)
                    canvas.blit(time_surface, (int(int_x) + marker_size + 3, int(int_y) + 15))
                    
                    cr_text = f"cr:{closing_rate:+.2f}m/s"
                    cr_surface = cpa_font.render(cr_text, True, obs_color)
                    canvas.blit(cr_surface, (int(int_x) + marker_size + 3, int(int_y) + 30))

                    min_sep_text = f"sep:{min_separation:.2f}NM"
                    min_sep_surface = cpa_font.render(min_sep_text, True, obs_color)
                    canvas.blit(min_sep_surface, (int(int_x) + marker_size + 3, int(int_y) + 45))

                    if time_to_min_sep > 0 and time_to_min_sep < LONG_CONFLICT_THRESHOLD_SEC:
                        if hasattr(active_agent, 'cpa_position_map') and intruder_id in active_agent.cpa_position_map:
                            cpa_lat, cpa_lon = active_agent.cpa_position_map[intruder_id]
                            
                            if cpa_lat is not None and cpa_lon is not None:
                                active_lat, active_lon, _, _ = self.sim.traf_get_state(active_agent.id)
                                active_x, active_y = self.camera.latlon_to_screen(self.sim, active_lat, active_lon, self.window_width, self.window_height)
                                cpa_x, cpa_y = self.camera.latlon_to_screen(self.sim, cpa_lat, cpa_lon, self.window_width, self.window_height)
                                
                                if cpa_x is not None and cpa_y is not None and active_x is not None and active_y is not None:
                                    if 0 <= cpa_x <= self.window_width and 0 <= cpa_y <= self.window_height:
                                        cpa_point_color = (obs_color[0], obs_color[1], obs_color[2])
                                        pygame.draw.circle(canvas, cpa_point_color, (int(cpa_x), int(cpa_y)), 6, 2)
                                        pygame.draw.line(canvas, cpa_point_color, (int(active_x), int(active_y)), (int(cpa_x), int(cpa_y)), 1)
                else:
                    logger.debug(f"[RENDER] Slot {slot_idx} ({intruder_id}): NO collision_info in cache!")

        # Draw aircraft
        for agent in self.all_agents.get_all_agents():
            lat, lon, hdg, tas = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(self.sim, lat, lon, self.window_width, self.window_height)
            
            if x_pos is None or y_pos is None:
                continue
            
            if not (0 <= x_pos <= self.window_width and 0 <= y_pos <= self.window_height):
                continue

            if agent == self.all_agents.get_active_agent():
                color = (255,255,0)  # GELB für aktiven Agenten
            else: 
                color = (80,80,80)  # GRAU für passive Agenten

            line_length_km = 12 / NM2KM
            lat_end, lon_end = fn.get_point_at_distance(lat, lon, line_length_km, hdg)
            heading_x, heading_y = self.camera.latlon_to_screen(self.sim, lat_end, lon_end, self.window_width, self.window_height)
            pygame.draw.line(canvas, (0,0,0), (x_pos,y_pos), (heading_x,heading_y), 2)
            
            # Multi-heading CPA visualization
            if agent == self.all_agents.get_active_agent() and MULTI_CAP_HEADING_RENDER:
                obs_dict = agent.obs_dict_cache if agent.obs_dict_cache is not None else self._get_observation_dict(agent)
                multi_heading_cpa = obs_dict.get('multi_heading_cpa', None)
                
                if multi_heading_cpa is not None and len(multi_heading_cpa) == NUM_HEADING_OFFSETS:
                    fan_radius_nm = 30.0
                    fan_radius_km = fan_radius_nm * NM2KM
                    fan_radius_px = (fan_radius_km / self.camera.zoom_km) * self.window_width
                    
                    segment_half_angle = 5.0
                    fan_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                    
                    for offset_idx in range(NUM_HEADING_OFFSETS):
                        offset_deg = HEADING_OFFSETS[offset_idx]
                        cpa_value = multi_heading_cpa[offset_idx]
                        
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
                        
                        heading_with_offset = hdg + offset_deg
                        angle_left = heading_with_offset - segment_half_angle
                        angle_right = heading_with_offset + segment_half_angle
                        
                        lat_left, lon_left = fn.get_point_at_distance(lat, lon, fan_radius_km, angle_left)
                        lat_right, lon_right = fn.get_point_at_distance(lat, lon, fan_radius_km, angle_right)
                        
                        x_left, y_left = self.camera.latlon_to_screen(self.sim, lat_left, lon_left, self.window_width, self.window_height)
                        x_right, y_right = self.camera.latlon_to_screen(self.sim, lat_right, lon_right, self.window_width, self.window_height)
                        
                        if x_left is not None and y_left is not None and x_right is not None and y_right is not None:
                            pygame.draw.polygon(fan_surface, segment_color, 
                                              [(int(x_pos), int(y_pos)), 
                                               (int(x_left), int(y_left)), 
                                               (int(x_right), int(y_right))], 0)
                            
                            pygame.draw.line(fan_surface, (0, 0, 0, 80), (int(x_pos), int(y_pos)), (int(x_left), int(y_left)), 1)
                            pygame.draw.line(fan_surface, (0, 0, 0, 80), (int(x_pos), int(y_pos)), (int(x_right), int(y_right)), 1)
                            pygame.draw.line(fan_surface, (0, 0, 0, 80), (int(x_left), int(y_left)), (int(x_right), int(y_right)), 1)
                    
                    canvas.blit(fan_surface, (0, 0))
 
            dist_km = INTRUSION_DISTANCE * NM2KM / 2  
            radius_px = (dist_km / self.camera.zoom_km) * self.window_width
            if radius_px > 1:
                pygame.draw.circle(canvas, color, (int(x_pos), int(y_pos)), int(radius_px), 2)

            font = pygame.font.SysFont(None, 20)
            tas_text = f"{tas:.0f}m/s"
            text_surface = font.render(tas_text, True, (0, 0, 0))
            text_x = x_pos + radius_px + 5
            text_y = y_pos - 10
            canvas.blit(text_surface, (int(text_x), int(text_y)))

        # Time display
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
        one_nm_km = 1.0
        one_nm_px = (one_nm_km / self.camera.zoom_km) * self.window_width
        
        legend_bottom = self.window_height - 50
        legend_left = 20
        
        pygame.draw.line(canvas, (0, 0, 0), 
                        (legend_left, legend_bottom), 
                        (legend_left + one_nm_px, legend_bottom), 
                        width=3)
        
        pygame.draw.line(canvas, (0, 0, 0), 
                        (legend_left, legend_bottom - 5), 
                        (legend_left, legend_bottom + 5), 
                        width=2)
        pygame.draw.line(canvas, (0, 0, 0), 
                        (legend_left + one_nm_px, legend_bottom - 5), 
                        (legend_left + one_nm_px, legend_bottom + 5), 
                        width=2)
        
        legend_font_small = pygame.font.SysFont(None, 16)
        text_surface = legend_font_small.render("1 NM", True, (0, 0, 0))
        text_x = legend_left + (one_nm_px / 2) - (text_surface.get_width() / 2)
        text_y = legend_bottom + 10
        canvas.blit(text_surface, (text_x, text_y))


        self._render_debug_info(canvas, active_agent)

        # Pause overlay
        if self.is_paused:
            font_pause = pygame.font.SysFont(None, 48, bold=True)
            pause_text = font_pause.render("PAUSED - Press SPACE to resume", True, (255, 0, 0))
            text_rect = pause_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
            pygame.draw.rect(canvas, (255, 255, 255), text_rect.inflate(20, 20), border_radius=10)
            canvas.blit(pause_text, text_rect)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
        
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
