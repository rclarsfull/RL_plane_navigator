from typing import Tuple
import logging
import time
import functools
from collections import defaultdict
import numpy as np

import simulator.simulator_interface as simulator_interface

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

logger = logging.getLogger("AdapterLogger")
logger.setLevel(logging.INFO) 
handler = logging.FileHandler("AdapterLogger.log", mode="w", encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Global dictionary to track accumulated function times
function_times = defaultdict(float)

def time_function(func):
    """Decorator to track accumulated time spent in functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            function_name = f"{func.__module__}.{func.__qualname__}"
            function_times[function_name] += execution_time
            logger.debug(f"Function {function_name}: +{execution_time:.6f}s (total: {function_times[function_name]:.6f}s)")
    return wrapper

def log_function_times():
    """Log all accumulated function times"""
    logger.info("=== Accumulated Function Times ===")
    for func_name, total_time in sorted(function_times.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{func_name}: {total_time:.6f}s")
    logger.info("=== End Function Times ===")

def reset_function_times():
    """Reset all accumulated function times"""
    function_times.clear()
    logger.info("Function times reset")


class Simulator(simulator_interface.SimulatorInterface):

    # ===== Initialization =====
    @time_function
    def init(self, simulator_step_size: float = 1.0) -> None:
        bs.init(mode='sim', detached=True, discoverable=False)
        bs.scr = ScreenDummy()
        bs.stack.stack(f"DT {simulator_step_size};FF") 
        bs.stack.stack("RESO OFF")

    # ===== Traffic Management =====
    @time_function
    def traf_reset(self) -> None:
        bs.traf.reset()

    @time_function
    def traf_create(self, acid: str, actype: str, acspd: float,
                    aclat: float, aclon: float, achdg: float, acalt: int):
        # Validate heading before creating aircraft
        try:
            achdg = float(achdg)
        except (ValueError, TypeError) as e:
            logger.error(f"CRE {acid}: Cannot convert heading to float: {achdg}. Error: {e}. SKIPPING.")
            return
        
        if not np.isfinite(achdg):
            logger.error(f"CRE {acid}: Invalid heading (NaN/Inf): {achdg}. SKIPPING.")
            return
        
        cas_kts = acspd * 1.94384
        bs.stack.stack(f"CRE {acid} {actype} {aclat} {aclon} {achdg} FL{acalt} {cas_kts}")
        bs.stack.stack(f'RESOOFF {acid}')

    @time_function
    def traf_get_state(self, id: str) -> Tuple[float, float, float, float]:
        ac_idx = bs.traf.id2idx(id)
        if ac_idx < 0:
            raise ValueError(f"Aircraft {id} not found")
        hdg = bs.traf.hdg[ac_idx]
        lat = bs.traf.lat[ac_idx]
        lon = bs.traf.lon[ac_idx]
        tas = bs.traf.tas[ac_idx]
        return lat, lon, hdg, tas
    
    @time_function
    def traf_get_all(self) -> Tuple:
        """
        Liefert alle Flugzeug-Positionen für Berechnungen.
        
        Returns:
            Tuple: (lat_array, lon_array, hdg_array, tas_array, id_list)
                  Arrays mit allen Flugzeug-Daten für Matrix-Berechnungen
        """
        lat_array = bs.traf.lat.copy()
        lon_array = bs.traf.lon.copy()
        hdg_array = bs.traf.hdg.copy()
        tas_array = bs.traf.tas.copy()
        id_list = bs.traf.id.copy()
        return lat_array, lon_array, hdg_array, tas_array, id_list

    @time_function
    def traf_set_heading(self, id: str, new_heading: float):
        # Convert to Python float and validate
        try:
            if isinstance(new_heading, (np.ndarray, list, tuple)):
                new_heading = float(np.asarray(new_heading).flat[0])
            else:
                new_heading = float(new_heading)
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"HDG {id}: Cannot convert to float: {new_heading} ({type(new_heading)}). Error: {e}. SKIPPING.")
            return
        
        # Check if finite
        if not np.isfinite(new_heading):
            logger.error(f"HDG {id}: Invalid heading (NaN/Inf): {new_heading}. SKIPPING.")
            return
        
        bs.stack.stack(f"HDG {id} {new_heading}")

    @time_function
    def traf_set_speed(self, id: str, new_speed: float):
        # Umrechnung von m/s in CAS-Knoten
        # 1 m/s = 1.94384 Knoten
        cas_kts = new_speed * 1.94384
        bs.stack.stack(f"SPD {id},{cas_kts}")

    @time_function
    def traf_set_altitude(self, id: str, new_altitude: float):
        bs.stack.stack(f"ALT {id} {new_altitude}")

    @time_function
    def traf_add_waypoint(self, id: str, lat: float, lon: float, altitude: int = None):
        """Fügt einen Wegpunkt zur Route des Flugzeugs hinzu.
        
        Args:
            id: Aircraft ID
            lat: Latitude des Wegpunkts
            lon: Longitude des Wegpunkts
            altitude: Flugfläche (optional), z.B. FL100 => 100
        """
        if altitude is not None:
            bs.stack.stack(f"ADDWPT {id} {lat} {lon} FL{altitude}")
        else:
            bs.stack.stack(f"ADDWPT {id} {lat} {lon}")

    @time_function
    def traf_activate_lnav(self, id: str):
        """Aktiviert die laterale Navigation (Autopilot für Wegpunkte).
        
        Args:
            id: Aircraft ID
        """
        bs.stack.stack(f"LNAV {id} ON")

    # ===== Simulation Control =====
    @time_function
    def sim_step(self, dt: float = None) -> None:
        """Führt einen Simulator-Schritt durch.
        
        In BlueSky ist dt NICHT die absolute Schrittdauer, sondern wird für 
        Echtzeit-Anpassung verwendet. Die echte Schrittdauer wird durch das 
        DT-Kommando gesetzt.
        
        Args:
            dt: Gewünschte Schrittdauer in Sekunden. Setzt das DT-Kommando und 
                führt dann einen Step durch. Wenn None, verwendet die aktuelle DT-Einstellung.
        """
        if dt is not None:
            # Setze die Simulator-Schrittgröße dynamisch
            bs.stack.stack(f"DT {dt}")
        
        # Führe einen Simulator-Schritt durch
        bs.sim.step()

    # ===== Geometry Tools =====
    @time_function
    def geo_calculate_direction_and_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
        heading, dist_nm = bs.tools.geo.kwikqdrdist(lat1, lon1, lat2, lon2)
        return heading, dist_nm
