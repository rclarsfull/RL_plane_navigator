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
from .helper_functions import compute_cpa_multi_heading_numba
from simulator.blue_sky_adapter import Simulator


class Cross_env(gym.Env):
    metadata = {
        "name": "merge_v1",
        "render_modes": ["rgb_array","human"],
         "render_fps": 120}

    def __init__(self, render_mode=None):

        if render_mode is not None:
            assert render_mode in self.metadata["render_modes"], f"Invalid render_mode {render_mode}"
        
        self.render_mode = render_mode
        self.step_limit = int(TIME_LIMIT / AGENT_INTERACTION_TIME)

        self.window_width = 1500
        self.window_height = 1000
        self.window_size = (self.window_width, self.window_height)

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()

        self.is_paused = False  
        self.resets_counter = 0

        self.all_agents = None
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
        self.cumulative_speed_stability_reward = 0.0


        # Observation Space Breakdown:
        # ego_state: 6 (heading_cos, heading_sin, speed, drift, v_sep, dist_to_wp)
        # threat_features: NUM_AC_STATE * 8 = 4 * 8 = 32
        # action_history: 5 (last_action_heading, action_continuous, action_age, turning_rate, last_action_speed)
        # multi_heading_cpa: NUM_HEADING_OFFSETS (min time_to_cpa for each heading offset) = 9
        # Total: 6 + 32 + 5 + 9 = 52
        obs_dim = 6 + NUM_AC_STATE * 8 + 5 + NUM_HEADING_OFFSETS
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = gym.spaces.MultiDiscrete([2, 180, 3])

        self.sim = Simulator()
        self.sim.init(simulator_step_size=1.0)

        self.total_reward = 0
        self.num_episodes = 0
        self.total_intrusions = 0

        self.center_lat = CENTER_LAT
        self.center_lon = CENTER_LON
        self.agent_trails = {}
        self.action_markers = {}
        
        self.camera = Camera(center_lat=self.center_lat, center_lon=self.center_lon, zoom_km=667)  

    
    def _is_actual_danger(self, closing_rate, time_to_min_sep, min_separation):
        """
        Prüft ob eine Situation kritisch/gefährlich ist.
        
        Unterstützt sowohl einzelne Werte (float) als auch NumPy-Arrays (vektorisiert).
        
        Returns:
            bool oder np.ndarray[bool]: True/True-Maske wo Gefahr besteht
        """
        is_closing = closing_rate < DANGER_CLOSING_THRESHOLD
        is_near_future = (time_to_min_sep > 0) & (time_to_min_sep < LONG_CONFLICT_THRESHOLD_SEC)
        is_dangerous = min_separation < DANGER_MIN_SEP_THRESHOLD

        return is_closing & is_near_future & is_dangerous

    def _calculate_collision_criticality_vectorized(self, agent_lat, agent_lon, agent_hdg, agent_tas,
                                                   intruder_lats, intruder_lons, intruder_hdgs, intruder_tas_array):

        is_debug = logger.isEnabledFor(logging.DEBUG)
        n_intruders = len(intruder_lats)
        
        if n_intruders == 0:
            return {}
        
        # Batch-Berechnung: Alle Bearings + Distanzen auf einmal (UNVERMEIDBAR - Simulator Call)
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

        # Extract column 0 (the 0° offset) -> shapes become [n_intruders]
        min_separation = min_seps_2d[:, 0]
        t_to_cpa = t_to_cpas_2d[:, 0]
        closing_rate = c_rates_2d[:, 0]

        # Recompute boolean flags (these were returned by the previous function)
        is_colliding = distances <= INTRUSION_DISTANCE
        flugbahnen_schneiden = min_separation <= INTRUSION_DISTANCE
        future_collision = (min_separation <= INTRUSION_DISTANCE * 1.5) & (closing_rate < 0) & (t_to_cpa > 0) & (t_to_cpa < 30.0)
        
        # Dictionary mit Ergebnissen
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

    def _calculate_cpa_multi_heading_vectorized(self, agent_lat, agent_lon, agent_hdg, agent_tas,
                                               intruder_lats, intruder_lons, intruder_hdgs, intruder_tas_array):
        n_intruders = len(intruder_lats)
        
        if n_intruders == 0:
            return {}
        
        # Batch-Berechnung: Alle Bearings + Distanzen auf einmal
        bearings = np.zeros(n_intruders, dtype=np.float64)
        distances = np.zeros(n_intruders, dtype=np.float64)
        
        for i in range(n_intruders):
            bearings[i], distances[i] = self.sim.geo_calculate_direction_and_distance(
                agent_lat, agent_lon, intruder_lats[i], intruder_lons[i]
            )

        min_seps, t_cpas, c_rates = compute_cpa_multi_heading_numba(
            agent_hdg, agent_tas,
            intruder_hdgs, intruder_tas_array,
            bearings, distances,
            INTRUSION_DISTANCE,
            HEADING_OFFSETS
        )
        
        # Strukturiere Results in gut lesbarem Dict
        results = {}
        for i in range(n_intruders):
            results[i] = {}
            for offset_idx in range(NUM_HEADING_OFFSETS):
                offset_deg = HEADING_OFFSETS[offset_idx]
                results[i][f'offset_{offset_idx}'] = {
                    'min_sep': float(min_seps[i, offset_idx]),
                    'time_to_cpa': float(t_cpas[i, offset_idx]),
                    'closing_rate': float(c_rates[i, offset_idx]),
                    'offset_deg': float(offset_deg)
                }
        
        del bearings, distances, min_seps, t_cpas, c_rates  # Free memory
        return results

    def _compute_multi_heading_cpa_features(self, agent: Agent, ac_lat: float, ac_lon: float, ac_hdg: float, ac_tas: float) -> np.ndarray:
        """
        VEKTORISIERTE Berechnung: Minimale time_to_cpa für jeden Heading-Offset (nur kritische Konflikte).
        
        Returns: np.ndarray mit NUM_HEADING_OFFSETS Werten (normalisiert auf [0, 1])
        - Wenn kein kritischer Konflikt: 1.0 (=sicher, weit weg in Zeit)
        - Wenn kritischer Konflikt: normalized time_to_cpa [0, 1]
        
        OPTIMIERUNG: Keine Python for-loops! Alles in NumPy-Arrays.
        """
        # Default: Alle Offsets auf 1.0 (=sicher)
        multi_heading_cpa_features = np.ones(NUM_HEADING_OFFSETS, dtype=np.float32)
        
        if not hasattr(agent, 'intruder_state_map') or len(agent.intruder_state_map) == 0:
            return multi_heading_cpa_features
        
        # Sammle alle Intruder-Daten (einmaliger Loop, unvermeidbar für Dict-Zugriff)
        intruder_data = list(agent.intruder_state_map.values())
        if len(intruder_data) == 0:
            return multi_heading_cpa_features
        
        intruder_lats = np.array([d[0] for d in intruder_data], dtype=np.float64)
        intruder_lons = np.array([d[1] for d in intruder_data], dtype=np.float64)
        intruder_hdgs = np.array([d[2] for d in intruder_data], dtype=np.float64)
        intruder_tas_array = np.array([d[3] for d in intruder_data], dtype=np.float64)
        
        n_intruders = len(intruder_lats)
        
        # Berechne Bearings + Distances (UNVERMEIDBAR - Simulator Call)
        bearings = np.zeros(n_intruders, dtype=np.float64)
        distances = np.zeros(n_intruders, dtype=np.float64)
        for i in range(n_intruders):
            bearings[i], distances[i] = self.sim.geo_calculate_direction_and_distance(
                ac_lat, ac_lon, intruder_lats[i], intruder_lons[i]
            )
        
        # ⚡ NUMBA-OPTIMIERTE MULTI-HEADING CPA-BERECHNUNG (returniert 2D-Arrays!)
        # Shape: [n_intruders, NUM_HEADING_OFFSETS]
        min_seps, t_cpas, c_rates = compute_cpa_multi_heading_numba(
            ac_hdg, ac_tas,
            intruder_hdgs, intruder_tas_array,
            bearings, distances,
            INTRUSION_DISTANCE,
            HEADING_OFFSETS
        )
        
        # ⚡ VEKTORISIERTE KRITIKALITÄTSPRÜFUNG mit _is_actual_danger()
        # Shape: [n_intruders, NUM_HEADING_OFFSETS] - True = kritischer Konflikt
        is_critical = self._is_actual_danger(c_rates, t_cpas, min_seps)
        
        # ⚡ VEKTORISIERTE MIN-BERECHNUNG: Für jeden Offset die minimale time_to_cpa
        # Trick: Setze nicht-kritische Konflikte auf inf, dann np.min pro Spalte
        t_cpas_masked = np.where(is_critical, t_cpas, np.inf)
        min_critical_times = np.min(t_cpas_masked, axis=0)  # Shape: [NUM_HEADING_OFFSETS]
        
        # ⚡ VEKTORISIERTE NORMALISIERUNG: Alle Offsets auf einmal
        valid_mask = min_critical_times < np.inf
        multi_heading_cpa_features[valid_mask] = min_critical_times[valid_mask] / LONG_CONFLICT_THRESHOLD_SEC
        
        # Debug-Logging (nur wenn aktiviert)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Step {self.steps}] MULTI-HEADING CPA (VEKTORISIERT) für {agent.id}:")
            for offset_idx in range(NUM_HEADING_OFFSETS):
                offset_deg = HEADING_OFFSETS[offset_idx]
                value = multi_heading_cpa_features[offset_idx]
                status = "SAFE" if value >= 0.99 else f"DANGER (t={value*LONG_CONFLICT_THRESHOLD_SEC:.1f}s)"
                logger.debug(f"  Offset {offset_deg:+.0f}°: {value:.3f} → {status}")
        
        # Cleanup
        del bearings, distances, min_seps, t_cpas, c_rates, is_critical, t_cpas_masked
        
        return multi_heading_cpa_features

    def _calculate_collision_criticality(self, agent_lat, agent_lon, agent_hdg, agent_tas, agent_turning_rate,
                                        intruder_lat, intruder_lon, intruder_hdg, intruder_tas, intruder_turning_rate):
        """
        Berechnet Kollisions-Rohwerte: min_separation, time_to_min_sep, closing_rate.
        OPTIMIERT: Nur essenzielle Numpy-Operationen, Logging-Overhead reduziert.
        """
        is_debug = logger.isEnabledFor(logging.DEBUG)
        
        bearing, distance = self.sim.geo_calculate_direction_and_distance(
            agent_lat, agent_lon, intruder_lat, intruder_lon
        )
        is_colliding = distance <= INTRUSION_DISTANCE
        
        if is_debug:
            logger.debug(f"\n[CAP-CALC START] Agent: ({agent_lat:.6f}, {agent_lon:.6f}), Intruder: ({intruder_lat:.6f}, {intruder_lon:.6f})")
            logger.debug(f"[CAP-CALC] Distance: {distance:.4f} NM, Bearing: {bearing:.2f}°")
        
        # Positionen als numpy arrays
        intruder_pos = np.array(self._bearing_distance_to_cartesian(bearing, distance))
        agent_pos = np.zeros(2)

        # Richtungsvektoren und Geschwindigkeiten
        agent_dir = np.array(self._get_direction_vector(agent_hdg))
        intruder_dir = np.array(self._get_direction_vector(intruder_hdg))
        
        agent_speed_nm_s = agent_tas / 1852.0
        intruder_speed_nm_s = intruder_tas / 1852.0
        
        agent_vel = agent_dir * agent_speed_nm_s
        intruder_vel = intruder_dir * intruder_speed_nm_s

        # Relative Bewegung
        rel_pos = intruder_pos - agent_pos
        rel_vel = intruder_vel - agent_vel
        
        # CPA-Berechnung: Alle Norms kombiniert berechnen
        rel_vel_mag_sq = np.dot(rel_vel, rel_vel)
        rel_pos_mag_sq = np.dot(rel_pos, rel_pos)
        rel_pos_mag = np.sqrt(rel_pos_mag_sq)
        
        if is_debug:
            logger.debug(f"[CAP-CALC] Agent: hdg={agent_hdg:.2f}°, tas={agent_tas:.2f}m/s → {agent_speed_nm_s:.6f}NM/s")
            logger.debug(f"[CAP-CALC] Intruder: hdg={intruder_hdg:.2f}°, tas={intruder_tas:.2f}m/s → {intruder_speed_nm_s:.6f}NM/s")
            logger.debug(f"[CAP-CALC] rel_pos: {rel_pos}, rel_vel: {rel_vel}, |rel_vel|={np.sqrt(rel_vel_mag_sq):.6f}")
        
        # Zeit zum CPA
        if rel_vel_mag_sq > 1e-6:
            dot_product = np.dot(rel_pos, rel_vel)
            t_to_cpa = -dot_product / rel_vel_mag_sq
            if is_debug:
                logger.debug(f"[CAP-CALC] dot(rel_pos, rel_vel)={dot_product:.6f}, t_to_cpa={t_to_cpa:.4f}s")
        else:
            t_to_cpa = 0.0
            if is_debug:
                logger.debug(f"[CAP-CALC] rel_vel too small (parallel courses), t_to_cpa=0.0s (instant CPA)")
        
        # Min Separation am CPA
        cpa_rel_pos = rel_pos + t_to_cpa * rel_vel
        min_separation = np.sqrt(np.dot(cpa_rel_pos, cpa_rel_pos))
        
        if is_debug:
            logger.debug(f"[CAP-CALC] cpa_rel_pos: {cpa_rel_pos}, min_separation={min_separation:.4f}NM")
        
        # Closing Rate (nur wenn rel_pos > 0.001)
        if rel_pos_mag > 0.001:
            relative_direction = rel_pos / rel_pos_mag
            closing_rate = float(np.dot(rel_vel, relative_direction))
        else:
            closing_rate = 0.0
        
        if is_debug:
            logger.debug(f"[CAP-CALC] closing_rate={closing_rate:.6f}NM/s (negative=approaching, positive=separating)")
        
        flugbahnen_schneiden = (min_separation <= INTRUSION_DISTANCE)
        future_collision = (min_separation <= INTRUSION_DISTANCE * 1.5) and (closing_rate < 0) and (t_to_cpa > 0) and (t_to_cpa < 30.0)
        
        if is_debug:
            logger.debug(f"[CAP-CALC RESULT] min_sep={min_separation:.4f}NM, time_to_sep={t_to_cpa:.4f}s, cr={closing_rate:.6f}, future_collision={future_collision}\n")
        
        return {
            'min_separation': float(min_separation),
            'time_to_min_sep': float(t_to_cpa),
            'closing_rate': float(closing_rate),
            'is_colliding': bool(is_colliding),
            'flugbahnen_schneiden': bool(flugbahnen_schneiden),
            'future_collision': bool(future_collision)
        }

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
        self.cumulative_speed_stability_reward = 0.0
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
    
    def _create_agents(self, n_agents, n_random_agents):
        """Erstellt Agent-Liste mit gegebener Größe."""
        agents = Agents()
        total_agents = n_agents + n_random_agents
        for i in range(total_agents):
            agents.add_agent(Agent(f'kl00{i+1}'.upper(), 0.0, False, 999.0, 0.0))
        return agents

    def _generate_waypoints(self, num_waypoints=4, center_lat=CENTER_LAT, center_lon=CENTER_LON, grid_size_km=300, rotation_deg=0):
        """Erstellt Waypoints auf einem Raster um die Zentrumsposition."""
        waypoints = deque()
        x_spacing = grid_size_km / num_waypoints
        rotation_rad = np.deg2rad(rotation_deg)
        cos_rot = np.cos(rotation_rad)
        sin_rot = np.sin(rotation_rad)
    
        for i in range(num_waypoints):
            grid_x = grid_size_km/2 - (i * x_spacing) - 200
            grid_y = random.uniform(-grid_size_km/4, grid_size_km/4)
            
            rotated_x = grid_x * cos_rot - grid_y * sin_rot
            rotated_y = grid_x * sin_rot + grid_y * cos_rot
            
            waypoint_lon = center_lon + (rotated_y / 111.32)
            waypoint_lat = center_lat + (rotated_x / 110.54)
            
            waypoints.append(Waypoint(waypoint_lat, waypoint_lon))
        return waypoints
    
    def _generate_crossing_routes(self, center_lat, center_lon, n_agents, n_random_agents):
        """Generiert Flugzeugrouten mit random Verteilung auf Ringe."""
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
                        diff = abs((candidate_angle - existing_angle + 180) % 360 - 180)
                        if diff < (angle_spacing * 0.8):
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
            ring_radius_nm = 100 + (assigned_ring * 15)  # 100, 115, 130, 145 NM
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
            
            # Calculate heading to FIRST waypoint
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
                'ring_idx': -1  # Marker für Random Agent
            }
        
        return routes_dict
    
    def _gen_aircraft(self):
        """Generiert Flugzeuge mit randomisierter Anzahl und Ring-Verteilung."""

        n_agents = random.randint(3, 7)
        n_random_agents = random.randint(0, 3)
        
        logger.debug(f"Episode #{self.num_episodes}: Spawning {n_agents} ring agents + {n_random_agents} random agents")
        

        self.all_agents = self._create_agents(n_agents, n_random_agents)
        

        routes_data = self._generate_crossing_routes(
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
            n_agents=n_agents,
            n_random_agents=n_random_agents
        )
        

        ring_stats = {}
        for route_info in routes_data.values():
            ring_idx = route_info['ring_idx']
            ring_stats[ring_idx] = ring_stats.get(ring_idx, 0) + 1
        
        if logger.isEnabledFor(logging.DEBUG):
            for ring_idx in sorted(ring_stats.keys()):
                if ring_idx == -1:
                    logger.debug(f"  Random Agents: {ring_stats[ring_idx]} (free positioning)")
                else:
                    radius_nm = 100 + (ring_idx * 15)
                    logger.debug(f"  Ring {ring_idx}: {ring_stats[ring_idx]} agents @ {radius_nm} NM")
        
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  Spawn collision for {agent.id}, applying offset")

                    offset_dist = random.uniform(0.5, 2.0)
                    offset_angle = random.uniform(0, 360)
                    start_lat, start_lon = fn.get_point_at_distance(
                        start_lat, start_lon, offset_dist, offset_angle
                    )
                    break
            
            spawn_positions.append((start_lat, start_lon))
            

            self.sim.traf_create(
                agent.id,
                actype="A320",
                acspd=speed + random.uniform(-5, 5),
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

    def _select_intruders_by_cpa(self, agent: Agent, ac_lat: float, ac_lon: float, ac_hdg: float, ac_tas: float, collision_info_cache: Dict) -> List[str]:
        """
        VEREINFACHTE CPA-SELEKTION: Nur echte Gefahren!
        
        Echte Gefahr = Konflikt-Potenzial:
        - closing_rate < 0 (näherkommen)
        - time_to_min_sep > 0 und < LONG_CONFLICT_THRESHOLD_SEC (zeitlich nah bevorstehend) ← HÖCHSTE PRIORITÄT!
        - min_separation < 20 NM (räumlich gefährlich)
        
        SORTIERUNG: Nach time_to_min_sep (kleinste Zeit = höchste Gefahr!)
        Ein Konflikt in 10 Sekunden ist gefährlicher als einer in 50 Sekunden, egal wie nah räumlich!
        
        Alle anderen = DUMMY-Daten (Agents verwirren sich nicht mit irrelevanten Flights!)
        """

        real_threats = []
        
        for intruder_id, collision_info in collision_info_cache.items():
            min_separation = collision_info['min_separation']
            closing_rate = collision_info['closing_rate']
            time_to_min_sep = collision_info['time_to_min_sep']
            
            if self._is_actual_danger(closing_rate, time_to_min_sep, min_separation):
                real_threats.append((intruder_id, time_to_min_sep))
        
        real_threats.sort(key=lambda x: x[1])
        selected_ids_cpa = [intruder_id for intruder_id, _ in real_threats[:NUM_AC_STATE]]
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n[Step {self.steps}] CPA SELECTION für {agent.id}")
            logger.debug(f"  Total Intruder: {len(collision_info_cache)}")
            logger.debug(f"  Real Threats (danger): {len(real_threats)}")
            logger.debug(f"  Selected (Top {NUM_AC_STATE}): {len(selected_ids_cpa)}")
            for idx, (intruder_id, min_sep) in enumerate(real_threats[:NUM_AC_STATE]):
                logger.debug(f"    Slot {idx}: {intruder_id} -> min_sep={min_sep:.2f}NM")
        
        return selected_ids_cpa



    def _compute_intruder_features(self, agent: Agent, ac_lat: float, ac_lon: float, ac_hdg: float, ac_tas: float) -> np.ndarray:

        FILLER_SLOT = [
            1.0,  # distance_normalized = 1.0 (weit weg ~150 NM = sicher)
            -1.0, 0.0,  # bearing_cos=-1.0, bearing_sin=0.0 (direkt HINTER = 180°)
            0.0,  # closing_rate_normalized = 0.0 (nicht näherkommen)
            1.0,  # min_separation_normalized = 1.0 (groß = safe, >10 NM)
            1.0,  # time_to_min_sep_normalized = 1.0 (weit weg in Zeit, >LONG_CONFLICT_THRESHOLD_SEC)
            -1.0, 0.0  # cpa_rel_x=-1.0 (weit hinter), cpa_rel_y=0.0 (zentral)
        ]
        intruder_features = np.array(FILLER_SLOT * NUM_AC_STATE, dtype=np.float32)
        

        closest_ids = agent.selected_intruder_ids
        

        if logger.isEnabledFor(logging.DEBUG) and len(closest_ids) < NUM_AC_STATE:
            num_fillers = NUM_AC_STATE - len(closest_ids)
            logger.debug(f"[Step {self.steps}] OBSERVATION SPACE für {agent.id}: "
                        f"{len(closest_ids)} echte Threats + {num_fillers} FILLER-SLOTS (alle mit SAFE-Defaults)")
        
        for idx, intruder_id in enumerate(closest_ids):
            if intruder_id not in agent.intruder_state_map:
                continue
                
            int_lat, int_lon, int_hdg, int_tas = agent.intruder_state_map[intruder_id]
            collision_info = agent.intruder_collision_cache[intruder_id]
            distance_nm = agent.intruder_distance_map[intruder_id]
            
            bearing, _ = self.sim.geo_calculate_direction_and_distance(ac_lat, ac_lon, int_lat, int_lon)
            bearing_diff = fn.bound_angle_positive_negative_180(bearing - ac_hdg)
            bearing_diff_rad = np.deg2rad(bearing_diff)
            bearing_cos = np.cos(bearing_diff_rad)
            bearing_sin = np.sin(bearing_diff_rad)
            
            closing_rate = collision_info['closing_rate']
            
            min_separation = collision_info['min_separation']
            

            time_to_min_sep = collision_info['time_to_min_sep']
            

            cpa_rel_x_normalized = 0.0
            cpa_rel_y_normalized = 0.0
            cpa_lat_cached = None
            cpa_lon_cached = None
            
            if time_to_min_sep > 0 and time_to_min_sep < LONG_CONFLICT_THRESHOLD_SEC:
                try:

                    active_vel_nm_s = ac_tas / 1852.0
                    active_distance_traveled = active_vel_nm_s * time_to_min_sep
                    agent_cpa_lat, agent_cpa_lon = fn.get_point_at_distance(
                        ac_lat, ac_lon, active_distance_traveled, ac_hdg
                    )
                    

                    intruder_vel_nm_s = int_tas / 1852.0
                    intruder_distance_traveled = intruder_vel_nm_s * time_to_min_sep
                    intruder_cpa_lat, intruder_cpa_lon = fn.get_point_at_distance(
                        int_lat, int_lon, intruder_distance_traveled, int_hdg
                    )
                    

                    cpa_lat_cached = (agent_cpa_lat + intruder_cpa_lat) / 2.0
                    cpa_lon_cached = (agent_cpa_lon + intruder_cpa_lon) / 2.0
                    

                    bearing_to_cpa, distance_to_cpa = self.sim.geo_calculate_direction_and_distance(
                        ac_lat, ac_lon, cpa_lat_cached, cpa_lon_cached
                    )
                    

                    bearing_to_cpa_rel = fn.bound_angle_positive_negative_180(bearing_to_cpa - ac_hdg)
                    bearing_to_cpa_rad = np.deg2rad(bearing_to_cpa_rel)
                    

                    cpa_rel_x = distance_to_cpa * np.cos(bearing_to_cpa_rad)
                    cpa_rel_y = distance_to_cpa * np.sin(bearing_to_cpa_rad)
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
                cpa_rel_x,
                cpa_rel_y
            ]
            
            base_idx = idx * 8
            intruder_features[base_idx:base_idx+8] = features_list
        

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Step {self.steps}] THREAT FEATURES für {agent.id}:")
            for slot_idx in range(min(3, NUM_AC_STATE)):  # Zeige erste 3 Slots
                slot_data = intruder_features[slot_idx*8:(slot_idx+1)*8]
                is_filler = np.allclose(slot_data, [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
                threat_type = "FILLER (SAFE)" if is_filler else f"THREAT"
                logger.debug(f"  Slot {slot_idx}: {threat_type} → {slot_data}")
        
        return intruder_features

    def _compute_action_history_refactored(self, agent: Agent) -> np.ndarray:
        """
        REFACTORED Action History (5 features):
        - last_action_do: [0, 1] - War letzte Aktion ein Lenken?
        - last_action_continuous: [-1, 1] - Lenkintensität
        - action_age_normalized: [0, 1] - Alter der Aktion (max 120 Steps = 10 Minuten)
        - turning_rate_normalized: [-1, 1] - Aktuelle Rotationsrate
        - last_action_speed: [0, 1, 2] - Normalisiert zu [-1, 0, 1] (0=Beschleunigen, 1=Nichts, 2=Verlangsamen)
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

    def _get_observation_dict(self, agent: Optional[Agent] = None) -> Dict[str, np.ndarray]:
        """
        REFACTORED: Observation Space mit Multi-Heading CPA Analyse
        
        Returns dict mit 5 Komponenten:
        - ego_state: (6) Heading, Speed, Drift, V-Sep, Distance to Waypoint
        - threat_features: (NUM_AC_STATE*8) Intruder Features (aktueller Heading)
        - action_history: (5) Last Action, Age, Turn Rate, Speed Action
        - multi_heading_cpa: (NUM_HEADING_OFFSETS) Min time_to_cpa für jeden Heading-Offset
        """
        if agent is None:
            agent = self.all_agents.get_active_agent()

        ac_lat, ac_lon, ac_hdg, ac_tas = agent.ac_lat, agent.ac_lon, agent.ac_hdg, agent.ac_tas
        
        # ===== EGO STATE =====
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
        
        # ===== THREAT FEATURES (aktueller Heading) =====
        threat_features = self._compute_intruder_features(agent, ac_lat, ac_lon, ac_hdg, ac_tas)
        
        # ===== MULTI-HEADING CPA FEATURES =====
        multi_heading_cpa = self._compute_multi_heading_cpa_features(agent, ac_lat, ac_lon, ac_hdg, ac_tas)
        
        # ===== ACTION HISTORY =====
        action_history = self._compute_action_history_refactored(agent)
        
        return {
            'ego_state': ego_state,
            'threat_features': threat_features,
            'multi_heading_cpa': multi_heading_cpa,
            'action_history': action_history
        }
         
    def _get_observation(self, agent: Optional[Agent] = None) -> np.ndarray:

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
            'cumulative_speed_stability_reward': float(self.cumulative_speed_stability_reward),
            'cumulative_noop_reward': float(self.cumulative_noop_reward),
            'avg_action_age': float(avg_action_age),
            'avg_action_age_seconds': float(avg_action_age_seconds),
            'total_cumulative_reward': float(self.cumulative_drift_reward + self.cumulative_intrusion_reward + 
                                              self.cumulative_action_age_reward + self.cumulative_cpa_warning_reward +
                                              self.cumulative_waypoint_bonus +
                                              self.cumulative_proximity_reward +
                                              self.cumulative_speed_stability_reward +
                                              self.cumulative_noop_reward),

            'last_reward_drift': float(last_reward_components.get('drift', 0.0)),
            'last_reward_action_age': float(last_reward_components.get('action_age', 0.0)),
            'last_reward_cpa_warning': float(last_reward_components.get('cpa_warning', 0.0)),
            'last_reward_proximity': float(last_reward_components.get('proximity', 0.0)),
            'last_reward_speed_stability': float(last_reward_components.get('speed_stability', 0.0)),
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
        5. speed_stability_reward: [0, SPEED_STABILITY_REWARD] (Belohnung für Geschwindigkeit NICHT ändern) ← JEDEN STEP!
        6. noop_reward: [NOOP_REWARD] (flache Belohnung für jede NOOP-Aktion) ← JEDEN STEP (wenn NOOP)!
        
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
        
        # === SPEED STABILITY REWARD ===
        # Belohnung dafür, die Geschwindigkeit NICHT zu ändern (last_action_speed == 1 = "Nichts")
        speed_stability_reward = 0.0
        if agent.last_action_speed == 1:
            speed_stability_reward = SPEED_STABILITY_REWARD
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Step {self.steps}] SPEED STABILITY REWARD: "
                        f"last_action_speed={agent.last_action_speed} | "
                        f"reward={speed_stability_reward:+.3f}")
        
        # === NOOP REWARD ===
        # Flache Belohnung für jede NOOP-Aktion (ermutigt Agent, weniger zu steuern)
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
                 speed_stability_reward +
                 noop_reward +
                 waypoint_bonus)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[Step {self.steps}] REWARDS: "
                        f"Drift={drift_reward:+.3f} | "
                        f"ActionAge={action_age_reward:+.3f} | "
                        f"CPA-Avoidance={collision_avoidance_reward:+.3f} | "
                        f"Proximity={proximity_reward:+.3f} | "
                        f"SpeedStability={speed_stability_reward:+.3f} | "
                        f"NOOP={noop_reward:+.3f} | "
                        f"TOTAL={reward:+.3f}")
        

        agent.last_reward_components = {
            'drift': float(drift_reward),
            'action_age': float(action_age_reward),
            'cpa_warning': float(collision_avoidance_reward),
            'proximity': float(proximity_reward),
            'speed_stability': float(speed_stability_reward),
            'noop': float(noop_reward),
            'total': float(reward)
        }
        
        self.cumulative_drift_reward += drift_reward
        self.cumulative_action_age_reward += action_age_reward
        self.cumulative_cpa_warning_reward += collision_avoidance_reward
        self.cumulative_proximity_reward += proximity_reward
        self.cumulative_speed_stability_reward += speed_stability_reward
        self.cumulative_noop_reward += noop_reward
        self.total_reward += reward
        return reward


    def _set_action(self, action, agent: Agent) -> None:
        """
        Verarbeitet MultiDiscrete Action: [do_action, steering_direction, speed_action]
        do_action: 0=No-Op, 1=Steering
        steering: 0-179 → [-1, 1] (180 Optionen = 2° pro Schritt)
        speed_action: 0=Beschleunigen (+D_SPEED), 1=Nichts, 2=Verlangsamen (-D_SPEED)
        """
        do_action = int(action[0])
        steering_index = int(action[1])
        speed_action = int(action[2])

        continuous_steering = (steering_index / 89.5) - 1.0
        continuous_steering = np.clip(continuous_steering, -1.0, 1.0)
        
        agent.last_action_continuous = continuous_steering
        agent.last_action_speed = speed_action
        self.last_continuous_action = continuous_steering
        
        is_active_agent = (agent == self.all_agents.get_active_agent())
        
        # OPTIMIZATION: Only track action markers if rendering mode is enabled
        if self.render_mode is not None:
            lat, lon, _, _ = self.sim.traf_get_state(agent.id)
            x_pos, y_pos = self.camera.latlon_to_screen(self.sim, lat, lon, self.window_width, self.window_height)
        else:
            x_pos, y_pos = None, None
        
        # Handle Heading Action
        if do_action == 0:
            agent.is_noop = True
            agent.last_action = 0
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
        elif do_action == 1:
            agent.is_noop = False
            agent.last_action = 1
            if is_active_agent:
                self.actions_steer_count += 1
            
            if self.render_mode is not None and x_pos is not None:
                if agent.id in self.action_markers:
                    self.action_markers[agent.id].append((x_pos, y_pos, 'steer'))
                    if len(self.action_markers[agent.id]) > MAX_ACTION_MARKERS:
                        self.action_markers[agent.id].pop(0)
                if hasattr(agent, 'action_markers_with_steering'):
                    agent.action_markers_with_steering.append((x_pos, y_pos, 'steer', continuous_steering, agent.action_age))
                    if len(agent.action_markers_with_steering) > MAX_ACTION_MARKERS:
                        agent.action_markers_with_steering.pop(0)
            
            dh = continuous_steering * D_HEADING
            _, _, current_heading, _ = self.sim.traf_get_state(agent.id)
            heading_new = fn.bound_angle_positive_negative_180(current_heading + dh)
            self.sim.traf_set_heading(agent.id, heading_new)
            agent.last_set_heading = heading_new
            agent.action_age = 0
        else:
            agent.is_noop = True
            agent.last_action = 0
        
        # Handle Speed Action (parallel zu Heading) - NUR wenn NICHT NOOP!
        # Bei NOOP (do_action == 0) wird NICHTS gemacht (weder Heading noch Speed)
        if do_action == 1:  # Nur bei aktiver Aktion (Steering)
            _, _, _, current_speed = self.sim.traf_get_state(agent.id)
            if speed_action == 0:
                # Beschleunigen
                new_speed = min(current_speed + D_SPEED, MAX_SPEED)
                self.sim.traf_set_speed(agent.id, new_speed)
            elif speed_action == 2:
                # Verlangsamen
                new_speed = max(current_speed - D_SPEED, MIN_SPEED)
                self.sim.traf_set_speed(agent.id, new_speed)
            # speed_action == 1: Nichts tun (Speed bleibt unverändert)

    def _render_frame(self):
        canvas = pygame.Surface(self.window_size)
        canvas.fill((169, 169, 169))  # Grauer Hintergrund (RGB)

        agent_positions = self.sim.traf_get_all()[:2]
        if len(agent_positions) > 0:
            self.camera.fixed_camera(CENTER_LAT, CENTER_LON)


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
            
            if agent.id in self.action_markers:
                for x_marker, y_marker, action_type in self.action_markers[agent.id]:
                    if 0 <= x_marker <= self.window_width and 0 <= y_marker <= self.window_height:
                        if action_type == 'steer':
                            pygame.draw.circle(canvas, (255, 50, 50), (int(x_marker), int(y_marker)), radius=5, width=0)
            
            if hasattr(agent, 'action_markers_with_steering') and agent.action_markers_with_steering:
                font_small = pygame.font.SysFont(None, 14)
                for x_marker, y_marker, action_type, steering_val, action_age in agent.action_markers_with_steering:
                    if 0 <= x_marker <= self.window_width and 0 <= y_marker <= self.window_height:
                        if action_type == 'steer':
                            action_age_seconds = action_age * AGENT_INTERACTION_TIME
                            action_age_minutes = action_age_seconds / 60.0
                            steering_text = f"{steering_val:.2f} (+{action_age_minutes:.1f}m)"
                            text_surface = font_small.render(steering_text, True, (0, 0, 0))
                            text_x = x_marker + 8
                            text_y = y_marker - 10
                            canvas.blit(text_surface, (int(text_x), int(text_y)))


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
                            pygame.draw.line(canvas, (0, 255, 0), (x1, y1), (x2, y2), 1)  # Dünne Linie
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
                

                obs_color = (255, 165, 0)  # Default: Orange (echte Gefahren in Obs Space)
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
                

                id_text = intruder_id[-3:]  # Letzte 3 Zeichen der ID
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
            
            # ===== MULTI-HEADING CPA VISUALISIERUNG (nur für aktiven Agenten) =====
            if agent == self.all_agents.get_active_agent() and MULTI_CAP_HEADING_RENDER:
                obs_dict = agent.obs_dict_cache if agent.obs_dict_cache is not None else self._get_observation_dict(agent)
                multi_heading_cpa = obs_dict.get('multi_heading_cpa', None)
                
                if multi_heading_cpa is not None and len(multi_heading_cpa) == NUM_HEADING_OFFSETS:
                    # Fächer-Visualisierung: Jeder Heading-Offset wird als Segment gezeichnet
                    fan_radius_nm = 30.0  # Radius des Fächers in NM
                    fan_radius_km = fan_radius_nm * NM2KM
                    fan_radius_px = (fan_radius_km / self.camera.zoom_km) * self.window_width
                    
                    segment_half_angle = 5.0  # Jedes Segment ist ±5° breit
                    
                    # Erstelle transparente Surface für Fächer-Segmente
                    fan_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                    
                    for offset_idx in range(NUM_HEADING_OFFSETS):
                        offset_deg = HEADING_OFFSETS[offset_idx]
                        cpa_value = multi_heading_cpa[offset_idx]  # [0, 1]: 1=sicher, 0=gefährlich
                        
                        # Farb-Interpolation: Rot (Gefahr) → Gelb (Warnung) → Grün (Sicher)
                        if cpa_value >= 0.8:
                            # Grün (sicher)
                            r = int(255 * (1.0 - cpa_value) / 0.2)
                            g = 255
                            b = 0
                        elif cpa_value >= 0.5:
                            # Gelb-Orange (Warnung)
                            r = 255
                            g = int(255 * (cpa_value - 0.5) / 0.3)
                            b = 0
                        else:
                            # Rot (Gefahr)
                            r = 255
                            g = 0
                            b = 0
                        
                        # Transparenz: Alpha-Kanal (120 = ca. 47% Deckkraft)
                        segment_color = (r, g, b, 120)
                        
                        # Berechne die beiden Randlinien des Segments
                        heading_with_offset = hdg + offset_deg
                        angle_left = heading_with_offset - segment_half_angle
                        angle_right = heading_with_offset + segment_half_angle
                        
                        # Endpunkte für die Segment-Linien
                        lat_left, lon_left = fn.get_point_at_distance(lat, lon, fan_radius_km, angle_left)
                        lat_right, lon_right = fn.get_point_at_distance(lat, lon, fan_radius_km, angle_right)
                        
                        x_left, y_left = self.camera.latlon_to_screen(self.sim, lat_left, lon_left, self.window_width, self.window_height)
                        x_right, y_right = self.camera.latlon_to_screen(self.sim, lat_right, lon_right, self.window_width, self.window_height)
                        
                        if x_left is not None and y_left is not None and x_right is not None and y_right is not None:
                            # Zeichne gefülltes Dreieck (Fächer-Segment) auf transparente Surface
                            pygame.draw.polygon(fan_surface, segment_color, 
                                              [(int(x_pos), int(y_pos)), 
                                               (int(x_left), int(y_left)), 
                                               (int(x_right), int(y_right))], 0)
                            
                            # Zeichne Rand des Segments (dünn, schwarz mit Transparenz)
                            pygame.draw.line(fan_surface, (0, 0, 0, 80), (int(x_pos), int(y_pos)), (int(x_left), int(y_left)), 1)
                            pygame.draw.line(fan_surface, (0, 0, 0, 80), (int(x_pos), int(y_pos)), (int(x_right), int(y_right)), 1)
                            pygame.draw.line(fan_surface, (0, 0, 0, 80), (int(x_left), int(y_left)), (int(x_right), int(y_right)), 1)
                    
                    # Blitte transparente Surface auf Canvas
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

        # Scale bar (1 NM) - unten links
        one_nm_km = 1.0  # 1 Nautical Mile
        one_nm_px = (one_nm_km / self.camera.zoom_km) * self.window_width
        
        legend_bottom = self.window_height - 50
        legend_left = 20
        legend_bar_height = 30
        

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

        # RENDER DEBUG: Zeichne Observation Space und Rewards auf den Canvas
        active_agent = self.all_agents.get_active_agent()
        if active_agent:
            obs_dict = active_agent.obs_dict_cache if active_agent.obs_dict_cache is not None else self._get_observation_dict(active_agent)
            reward_comp = active_agent.last_reward_components if hasattr(active_agent, 'last_reward_components') and active_agent.last_reward_components else {}
            info = self._get_info()
            
            # Fonts für verschiedene Größen
            font_title = pygame.font.SysFont(None, 20, bold=True)
            font_normal = pygame.font.SysFont(None, 16)
            font_small = pygame.font.SysFont(None, 14)
            
            debug_y = self.window_height - 250  # Nach unten verschieben
            debug_x_left = 10
            debug_x_right = self.window_width // 2 + 10
            
            # ==================== LINKE SEITE ====================
            # TITLE: Observation Space
            title_text = f"OBS SPACE - {active_agent.id} (Step {self.steps})"
            text_surf = font_title.render(title_text, True, (0, 0, 0))
            canvas.blit(text_surf, (debug_x_left, debug_y))
            debug_y += 25
            
            # Ego State
            ego_state = obs_dict['ego_state']
            text = f"EGO: Hdg({ego_state[0]:+.2f},{ego_state[1]:+.2f}) Spd({ego_state[2]:+.2f}) Drift({ego_state[3]:+.2f}) VrtSep({ego_state[4]:+.2f}) DstWp({ego_state[5]:+.2f})"
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
                slot_start = slot_idx * 8
                slot_data = threat_features[slot_start:slot_start+8]
                
                intruder_id = active_agent.selected_intruder_ids[slot_idx] if slot_idx < len(active_agent.selected_intruder_ids) else "FILLER"
                is_filler = np.allclose(slot_data, [1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0], atol=0.05)
                slot_type = "FILL" if is_filler else "THREAT"
                color = (100, 100, 100) if is_filler else (255, 100, 0)
                
                text = f"  [{slot_idx}]{intruder_id}({slot_type}): Dst({slot_data[0]:+.2f}) Brg({slot_data[1]:+.2f},{slot_data[2]:+.2f}) CR({slot_data[3]:+.2f}) MinSep({slot_data[4]:+.2f}) T({slot_data[5]:+.2f}) CPA({slot_data[6]:+.2f},{slot_data[7]:+.2f})"
                text_surf = font_small.render(text, True, color)
                canvas.blit(text_surf, (debug_x_left, debug_y))
                debug_y += 16
            
            # Action History
            action_history = obs_dict['action_history']
            text = f"ACTION: {'STEER' if action_history[0] > 0.5 else 'NOOP'} Steer({action_history[1]:+.2f}) Age({action_history[2]:+.2f}) TurnRate({action_history[3]:+.2f})"
            text_surf = font_small.render(text, True, (0, 100, 200))
            canvas.blit(text_surf, (debug_x_left, debug_y))
            debug_y += 18
            
            # ==================== RECHTE SEITE ====================
            debug_y_right = 10
            
            # TITLE: Rewards
            title_text = "REWARDS"
            text_surf = font_title.render(title_text, True, (0, 0, 0))
            canvas.blit(text_surf, (debug_x_right, debug_y_right))
            debug_y_right += 25
            
            # Reward Komponenten
            drift_r = reward_comp.get('drift', 0.0)
            action_age_r = reward_comp.get('action_age', 0.0)
            cpa_warning_r = reward_comp.get('cpa_warning', 0.0)
            proximity_r = reward_comp.get('proximity', 0.0)
            total_r = reward_comp.get('total', 0.0)
            
            text = f"Drift:        {drift_r:+.4f}"
            color = (0, 200, 0) if drift_r > 0 else (200, 0, 0)
            text_surf = font_normal.render(text, True, color)
            canvas.blit(text_surf, (debug_x_right, debug_y_right))
            debug_y_right += 20
            
            text = f"ActionAge:    {action_age_r:+.4f}"
            color = (0, 200, 0) if action_age_r > 0 else (200, 0, 0)
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
            
            # Total Reward (fett und groß)
            text = f"TOTAL: {total_r:+.4f}"
            color = (0, 255, 0) if total_r > 0 else (255, 0, 0)
            text_surf = font_title.render(text, True, color)
            canvas.blit(text_surf, (debug_x_right, debug_y_right))
            debug_y_right += 28
            
            # Episode Stats
            debug_y_right += 5
            text = f"Ep: {info['num_episodes']} | Intrusions: {info['intrusions']}"
            text_surf = font_small.render(text, True, (0, 0, 0))
            canvas.blit(text_surf, (debug_x_right, debug_y_right))
            debug_y_right += 18
            
            text = f"NoOp: {info['actions_noop']} | Steer: {info['actions_steer']}"
            text_surf = font_small.render(text, True, (0, 0, 0))
            canvas.blit(text_surf, (debug_x_right, debug_y_right))
            debug_y_right += 18
            
            text = f"Waypoints: {info['waypoints_collected']} | Avg Reward: {info['avg_reward']:+.3f}"
            text_surf = font_small.render(text, True, (0, 0, 0))
            canvas.blit(text_surf, (debug_x_right, debug_y_right))

        # Zeichne Pause-Text wenn Simulation pausiert
        if self.is_paused:
            font_pause = pygame.font.SysFont(None, 48, bold=True)
            pause_text = font_pause.render("PAUSED - Press SPACE to resume", True, (255, 0, 0))
            text_rect = pause_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
            pygame.draw.rect(canvas, (255, 255, 255), text_rect.inflate(20, 20), border_radius=10)
            canvas.blit(pause_text, text_rect)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

        # Event handling mit Pause-Feature
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Leertaste
                    self.is_paused = not self.is_paused
        
        # Wenn pausiert, warte auf Input (ignoriere FPS limit)
        while self.is_paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # Leertaste zum Fortsetzen
                        self.is_paused = False
            pygame.time.wait(100)  # CPU nicht überlasten während Pause

    def close(self):
        pygame.quit()
        
    def render(self, mode="human"):
         self.render_mode = mode
         if mode == "human":
             self._render_frame()
             time.sleep(0.05)
         elif mode == "rgb_array":
             logger.error("RGB array rendering not implemented")
             return np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

    def getNextWaypoint(self, agent: Agent) -> Waypoint:
        """Returns next waypoint for agent."""
        if len(agent.waypoint_queue) == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Agent {agent.id} has no waypoints!")
            return Waypoint(self.center_lat, self.center_lon)
        return agent.waypoint_queue[0]
    