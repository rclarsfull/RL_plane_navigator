import logging
import numpy as np

TARGET_DISTANCE_MARGIN = 5  # NM
INTRUSION_DISTANCE = 5  # NM

CPA_WARNING_FACTOR = 15.0 #20
LONG_CONFLICT_THRESHOLD_SEC = 600.0  
DRIFT_FACTOR = 1  
ACTION_AGE_FACTOR = 0.25  
PROXIMITY_REWARD_BASE = 0.0#3.0  
WAYPOINT_BONUS = 40.0

NUM_AC_STATE = 4  

SPAWN_SEPARATION_MIN = INTRUSION_DISTANCE * 1.2

AGENT_INTERACTION_TIME = 15 #sec
TIME_LIMIT = 60 * 60    # 1 hour

D_HEADING = 180
D_SPEED = 10  # m/s speed change per action (±10 m/s)
SPEED_STABILITY_REWARD = 0.05  # Reward für nicht-ändern der Geschwindigkeit
NOOP_REWARD = 0.2  # Flat reward for every NOOP action
FLIGHT_LEVEL = 245
MAX_SPEED = 150
MIN_SPEED = 125


DANGER_CLOSING_THRESHOLD = 0.0  # m/s: closing_rate < 0 = closing
DANGER_MIN_SEP_THRESHOLD = 7  # NM: max safe separation
OBS_DISTANCE = 100.0

CENTER_LAT = 49.9915
CENTER_LON = 8.6634
NM2KM = 1.852

# MEMORY OPTIMIZATION: Max lengths für Render-Caches
MAX_ACTION_MARKERS = 500  # Max Einträge in action_markers und action_markers_with_steering
MAX_AGENT_TRAILS = 1000  # Max Positionen in agent_trails

# Heading offsets für Multi-Heading CPA Analyse
HEADING_OFFSETS = np.array([-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64)
NUM_HEADING_OFFSETS = len(HEADING_OFFSETS)
MULTI_CAP_HEADING_RENDER = True

logger = logging.getLogger("Cross_env")



