from dataclasses import dataclass, field
import numpy as np
from typing import Deque, Dict, List, Optional, Any, Tuple
from .consts import NM2KM, NUM_AC_STATE, AGENT_INTERACTION_TIME, MIN_SPEED, MAX_SPEED, TARGET_DISTANCE_MARGIN, INTRUSION_DISTANCE
import bluesky_gym.envs.common.functions as fn


#from custom_envs.constants import *

@dataclass
class Waypoint:
    lat: float
    lon: float

@dataclass
class Agent:
    # Required fields
    id: str
    selected_heading: float
    is_finished: bool
    distanceTarget: float
    drift: float
    
    # Action tracking
    last_action: float = 0.0
    last_action_continuous: float = 0.0
    last_action_speed: int = 1  # 0=Beschleunigen, 1=Nichts, 2=Verlangsamen (Default: Nichts)
    last_action_type: str = 'noop'  # 'noop' | 'steer' | 'snap'
    is_noop: bool = False
    counter_no_op: int = 0
    action_markers_with_steering: List[Tuple] = field(default_factory=list)
    intrusions_caused_by_last_action: int = 0
    min_ttc: float = 500.0
    tti_to_intruders: Optional[List[float]] = None
    last_set_heading: float = 0.0
    heading_error: float = 0.0
    prev_drift: float = 0.0
    action_age: int = 0
    turning_rate: float = 0.0
    target_heading: Optional[float] = None  # Absolutes Ziel-Heading des Agents
    
    # Waypoint tracking
    waypoint_reached_this_step: bool = False
    waypoints_collected: int = 0
    waypoint_queue: Deque[Waypoint] = field(default_factory=Deque)
    
    # Reward tracking
    last_reward_components: Optional[Dict[str, float]] = None
    
    # Aircraft state (cached from simulator)
    ac_lat: float = 0.0
    ac_lon: float = 0.0
    ac_hdg: float = 0.0
    ac_hdg_prev: float = 0.0
    ac_tas: float = 0.0
    cos_drift: float = 0.0
    sin_drift: float = 0.0
    
    # Normalized features (cached from _update_observation_cache)
    speed: float = 0.0
    drift: float = 0.0
    distance_to_waypoint: float = 0.0
    
    # Observation caching
    obs_dict_cache: Optional[Dict[str, np.ndarray]] = None
    
    # Intruder tracking (cached in _update_observation_cache)
    intruder_collision_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    intruder_distance_map: Dict[str, float] = field(default_factory=dict)
    intruder_state_map: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    intruder_agent_map: Dict[str, Optional['Agent']] = field(default_factory=dict)
    selected_intruder_ids: List[str] = field(default_factory=list)
    cpa_position_map: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
    
    def __post_init__(self):
        """Initialize dynamic fields that might not be set via constructor."""
        if self.action_markers_with_steering is None:
            self.action_markers_with_steering = []
        if self.tti_to_intruders is None:
            self.tti_to_intruders = [500.0] * NUM_AC_STATE
        if self.last_reward_components is None:
            self.last_reward_components = {}
        if self.cpa_position_map is None:
            self.cpa_position_map = {}

    def update_observation_cache(self, env) -> None:
        """Move the heavy observation cache update into the Agent.

        This method mirrors the former environment implementation but operates
        on the agent instance and accepts the environment as `env` so it can
        access simulator, selection and calculation helpers.
        """
        # get simulator state for this agent
        ac_lat, ac_lon, ac_hdg, ac_tas = env.sim.traf_get_state(self.id)

        hdg_diff = ac_hdg - self.ac_hdg_prev
        hdg_diff = fn.bound_angle_positive_negative_180(hdg_diff)
        turn_rate_deg_per_sec = hdg_diff / AGENT_INTERACTION_TIME
        self.turning_rate = np.clip(turn_rate_deg_per_sec / 5.0, -1.0, 1.0)

        self.ac_lat = ac_lat
        self.ac_lon = ac_lon
        self.ac_hdg = ac_hdg
        self.ac_hdg_prev = ac_hdg
        self.ac_tas = ac_tas

        waypoint = env.getNextWaypoint(self)
        target_dir, target_dist = env.sim.geo_calculate_direction_and_distance(ac_lat, ac_lon, waypoint.lat, waypoint.lon)

        drift = ac_hdg - target_dir
        drift = fn.bound_angle_positive_negative_180(drift)
        drift_rad = np.deg2rad(drift)

        self.cos_drift = np.cos(drift_rad)
        self.sin_drift = np.sin(drift_rad)
        self.distanceTarget = target_dist
        self.distance_to_waypoint = target_dist
        self.drift = abs(drift_rad)
        self.speed = ac_tas

        heading_error = ac_hdg - self.last_set_heading
        heading_error = fn.bound_angle_positive_negative_180(heading_error)
        self.heading_error = heading_error

        self.prev_drift = self.drift

        # Gather all traffic from simulator
        all_lat, all_lon, all_hdg, all_tas, all_id = env.sim.traf_get_all()

        agent_id_map = {agent_obj.id: agent_obj for agent_obj in env.all_agents.get_all_agents()}

        # Reset intruder caches
        self.intruder_collision_cache.clear()
        self.intruder_distance_map.clear()
        self.intruder_state_map.clear()
        self.intruder_agent_map.clear()

        # Vectorized filtering of intruders
        all_lat_arr = np.array(all_lat)
        all_lon_arr = np.array(all_lon)
        all_hdg_arr = np.array(all_hdg)
        all_tas_arr = np.array(all_tas)
        all_id_arr = np.array(all_id)

        valid_mask = all_id_arr != self.id
        intruder_indices = np.where(valid_mask)[0]
        del valid_mask

        if len(intruder_indices) > 0:
            intruder_lats = all_lat_arr[intruder_indices]
            intruder_lons = all_lon_arr[intruder_indices]
            intruder_hdgs = all_hdg_arr[intruder_indices]
            intruder_tas_vec = all_tas_arr[intruder_indices]
            intruder_ids = all_id_arr[intruder_indices]

            collision_results_by_idx = env._calculate_collision_criticality_vectorized(
                ac_lat, ac_lon, ac_hdg, ac_tas,
                intruder_lats, intruder_lons, intruder_hdgs, intruder_tas_vec
            )
            del intruder_lats, intruder_lons, intruder_hdgs, intruder_tas_vec, intruder_ids

            # compute turning rates and populate state/agent maps
            turning_rates = np.zeros(len(intruder_indices))
            for idx, intruder_idx in enumerate(intruder_indices):
                intruder_id = all_id[intruder_idx]
                intruder_agent = agent_id_map.get(intruder_id)

                self.intruder_state_map[intruder_id] = (
                    all_lat[intruder_idx], all_lon[intruder_idx],
                    all_hdg[intruder_idx], all_tas[intruder_idx]
                )
                self.intruder_agent_map[intruder_id] = intruder_agent

                if intruder_agent is not None:
                    hdg_diff_intruder = all_hdg[intruder_idx] - intruder_agent.ac_hdg_prev
                    hdg_diff_intruder = fn.bound_angle_positive_negative_180(hdg_diff_intruder)
                    int_turning_rate_deg_per_sec = hdg_diff_intruder / AGENT_INTERACTION_TIME
                    turning_rates[idx] = np.clip(int_turning_rate_deg_per_sec / 5.0, -1.0, 1.0)

            del turning_rates

            # store collision info and distance
            for idx, intruder_idx in enumerate(intruder_indices):
                intruder_id = all_id[intruder_idx]
                collision_info = collision_results_by_idx[idx]
                self.intruder_collision_cache[intruder_id] = collision_info

                _, distance = env.sim.geo_calculate_direction_and_distance(
                    ac_lat, ac_lon, all_lat[intruder_idx], all_lon[intruder_idx]
                )
                self.intruder_distance_map[intruder_id] = distance

            del collision_results_by_idx

        # cleanup temporaries
        del all_lat_arr, all_lon_arr, all_hdg_arr, all_tas_arr, all_id_arr, intruder_indices, agent_id_map

        selected_ids_cpa = env._select_intruders_by_cpa(self, ac_lat, ac_lon, ac_hdg, ac_tas, self.intruder_collision_cache)

        self.selected_intruder_ids = selected_ids_cpa[:NUM_AC_STATE]

        # finally compute observation dict cache via environment helper
        self.obs_dict_cache = env._get_observation_dict(self)

    def check_finish(self, env) -> None:
        """Check if this agent reached its next waypoint (moved from env).

        Operates on the agent instance and uses the environment to query
        simulator state and waypoint management.
        """
        waypoint = env.getNextWaypoint(self)
        lat, lon, _, _ = env.sim.traf_get_state(self.id)
        _, self.distanceTarget = env.sim.geo_calculate_direction_and_distance(lat, lon, waypoint.lat, waypoint.lon)

        self.waypoint_reached_this_step = False
        if self.distanceTarget < TARGET_DISTANCE_MARGIN:
            self.waypoint_reached_this_step = True
            self.waypoints_collected += 1
            if len(self.waypoint_queue) > 0:
                self.waypoint_queue.popleft()

        if len(self.waypoint_queue) == 0:
            self.is_finished = True

    def check_intrusion(self, env) -> int:
        """Check intrusions for this agent (treating this as the active agent).

        Returns the number of intrusions detected. The environment's
        `total_intrusions` counter will be incremented when intrusions occur.
        """
        passive_agents = env.all_agents.get_passive_agents()
        if not passive_agents:
            return 0

        intruder_ids = [intruder.id for intruder in passive_agents]
        distances = np.array([self.intruder_distance_map.get(iid, np.inf) for iid in intruder_ids], dtype=np.float64)

        intrusion_mask = distances < INTRUSION_DISTANCE
        num_intrusions = int(np.sum(intrusion_mask))
        self.intrusions_caused_by_last_action = num_intrusions

        if num_intrusions > 0:
            env.total_intrusions += num_intrusions

            # Debug logging similar to original env implementation
            if getattr(env, 'logger', None) is not None and env.logger.isEnabledFor(env.logging.DEBUG if hasattr(env, 'logging') else 10):
                intrusion_indices = np.where(intrusion_mask)[0]
                first_intrusion_idx = int(intrusion_indices[0])
                intruder = passive_agents[first_intrusion_idx]
                int_dis = distances[first_intrusion_idx]

                obs_dict = self.obs_dict_cache
                threat_features = obs_dict['threat_features'] if obs_dict else None

                intruder_str = ""
                if threat_features is not None and len(threat_features) >= 6:
                    distance_norm = threat_features[0]
                    bearing_cos = threat_features[1]
                    bearing_sin = threat_features[2]
                    closing_rate_norm = threat_features[3]
                    min_sep_norm = threat_features[4]
                    time_to_min_sep_norm = threat_features[5]
                    intruder_str = (
                        f"\nRefactored Obs Space Threat Data:\n"
                        f"  Distance (norm): {distance_norm:.3f}\n"
                        f"  Bearing (cos/sin): ({bearing_cos:.3f}, {bearing_sin:.3f})\n"
                        f"  Closing Rate (norm): {closing_rate_norm:.3f}\n"
                        f"  Min Separation (norm): {min_sep_norm:.3f}\n"
                        f"  Time to Min Sep (norm): {time_to_min_sep_norm:.3f}"
                    )

                log_msg = (
                    f"\n{'='*80}\nINTRUSION DETECTED (AGENT-CHECK)\n{'='*80}\n"
                    f"Active: {self.id} (Hdg: {self.ac_hdg:.1f}°, Spd: {self.ac_tas:.1f} m/s)\n"
                    f"Intruder: {intruder.id} (Hdg: {intruder.ac_hdg:.1f}°, Spd: {intruder.ac_tas:.1f} m/s)\n"
                    f"Distance: {int_dis:.2f} NM (Threshold: {env.INTRUSION_DISTANCE} NM)\n"
                    f"Drift: {self.drift*180/np.pi:.1f}° | Steps: {getattr(env, 'steps', '?')} | Total Intrusions: {env.total_intrusions}"
                    f"{intruder_str}\n"
                    f"{'='*80}\n"
                )
                env.logger.debug(log_msg)

        return num_intrusions

@dataclass
class Camera:
    center_lat: float
    center_lon: float
    zoom_km: float = 1000
    
    def latlon_to_cartesian_nm(self, sim, lat: float, lon: float) -> Tuple[float, float]:
        """Konvertiert Lat/Lon zu kartesischen Koordinaten (NM) relativ zur Kamera."""
        qdr, dist = sim.geo_calculate_direction_and_distance(self.center_lat, self.center_lon, lat, lon)
        x_nm = dist * np.sin(np.deg2rad(qdr))
        y_nm = dist * np.cos(np.deg2rad(qdr))
        return x_nm, y_nm
    
    def latlon_to_screen(self, sim, lat: float, lon: float, window_width: int, window_height: int) -> Tuple[float, float]:
        """Wandelt Lat/Lon in Fensterkoordinaten um."""
        rel_x, rel_y = self.latlon_to_cartesian_nm(sim, lat, lon)
        screen_x = window_width / 2 + (rel_x * NM2KM / self.zoom_km) * window_width
        screen_y = window_height / 2 - (rel_y * NM2KM / self.zoom_km) * window_height
        
        # WICHTIG: Konvertiere zu native Python float statt numpy.float64!
        # pygame akzeptiert KEINE numpy types!
        return float(screen_x), float(screen_y)
    
    def center_on_agent(self, agent_lat: float, agent_lon: float) -> None:
        """Zentriere Kamera auf einen Agenten."""
        self.center_lat = agent_lat
        self.center_lon = agent_lon

    def fixed_camera(self, lat: float, lon: float) -> None:
        """Zentriere Kamera auf einen Punkt."""
        self.center_lat = lat
        self.center_lon = lon

    def center_on_centroid(self, agents_positions: List[Tuple[float, float]]) -> None:
        """Zentriere Kamera auf Schwerpunkt aller Agenten."""
        if not agents_positions:
            return
        avg_lat = sum(pos[0] for pos in agents_positions) / len(agents_positions)
        avg_lon = sum(pos[1] for pos in agents_positions) / len(agents_positions)
        self.center_lat = avg_lat
        self.center_lon = avg_lon
    
    def set_zoom_to_fit_all(self, agents_positions: List[Tuple[float, float]], window_width: int, window_height: int, margin: float = 1.2) -> None:
        """Setze Zoom um alle Agenten sichtbar zu machen."""
        if len(agents_positions) < 2:
            return
        max_dist = 0
        for i, pos1 in enumerate(agents_positions):
            for pos2 in agents_positions[i+1:]:
                _, dist = self.sim.geo_calculate_direction_and_distance(pos1[0], pos1[1], pos2[0], pos2[1])
                max_dist = max(max_dist, dist * NM2KM)
        self.zoom_km = max_dist * margin

class Agents:
    def __init__(self) -> None:
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)
    
    def get_all_agents(self) -> List[Agent]:
        return self.agents

    def get_active_agent(self) -> Agent:
        return self.agents[0]
    
    def get_passive_agents(self) -> List[Agent]:
        return self.agents[1:]

    def reset_all_agents(self) -> None:
        for agent in self.agents:
            agent.is_finished = False
            agent.drift = 0
            agent.last_action = 0.0
            agent.last_action_continuous = 0.0
            agent.last_action_speed = 1  # Reset to "Nichts"
            agent.is_noop = False
            agent.intrusions_caused_by_last_action = 0
            agent.counter_no_op = 0
            agent.min_ttc = 500.0
            agent.tti_to_intruders = [500.0] * NUM_AC_STATE  # Cache für TTC zu jedem Intruder
            agent.last_set_heading = 0.0
            agent.heading_error = 0.0
            agent.prev_drift = 0.0
            agent.action_age = 0
            agent.turning_rate = 0.0
            agent.waypoint_queue = Deque()
            agent.waypoints_collected = 0  # NEW: Reset waypoint counter
            agent.last_reward_components = {}  # NEW: Reset reward components
            agent.action_markers_with_steering = []
            # Reset Intruder caches
            agent.intruder_collision_cache = {}
            agent.intruder_distance_map = {}
            agent.intruder_state_map = {}
            agent.intruder_agent_map = {}
            agent.selected_intruder_ids = []

    def get_avg_drift(self) -> float:
        if not self.agents:
            return -1
        return sum(agent.drift for agent in self.agents) / len(self.agents)

    def is_active_agent_finished(self) -> bool:
        return self.agents[0].is_finished
