
from abc import ABC, abstractmethod
from typing import Tuple


class SimulatorInterface(ABC):
    """
    Minimales Interface für RL Training
    """
    # ===== Initialization =====
    @abstractmethod
    def init(self, simulator_step_size: float) -> None:
        """Initialisiert den Simulator"""
        raise NotImplementedError
    
    # ===== Traffic Management =====
    @abstractmethod
    def traf_reset(self) -> None:
        """Setzt den Verkehrsstatus zurück"""
        raise NotImplementedError
    
    @abstractmethod
    def traf_create(self, acid: str, actype: str, acspd: float, 
                   aclat: float, aclon: float, achdg: float, acalt: float) -> bool:
        """Erstellt ein neues Verkehrsobjekt"""
        raise NotImplementedError
    
    @abstractmethod
    def traf_get_state(self, id: str) -> Tuple[float, float, float, float]:
        """Liefert die Position (lat, lon, hdg, tas) eines Flugzeugs anhand der ID"""
        raise NotImplementedError
    
    @abstractmethod
    def traf_get_all(self) -> Tuple:
        """
        Liefert alle Flugzeug-Positionen für Berechnungen.
        
        Returns:
            Tuple: (lat_array, lon_array, hdg_array, tas_array, id_list)
                  Arrays mit allen Flugzeug-Daten für Matrix-Berechnungen
        """
        raise NotImplementedError

    @abstractmethod
    def traf_set_heading(self, id: str, new_heading: float):
        """Setzt die neue Heading eines Flugzeugs"""
        raise NotImplementedError

    @abstractmethod
    def traf_set_speed(self, id: str, new_speed: float):
        """Setzt die neue Geschwindigkeit eines Flugzeugs"""
        raise NotImplementedError

    @abstractmethod
    def traf_set_altitude(self, id: str, new_altitude: float):
        """Setzt die neue Flughöhe eines Flugzeugs"""
        raise NotImplementedError

    # ===== Simulation Control =====
    @abstractmethod
    def sim_step(self) -> None:
        """Entspricht bs.sim.step()"""
        raise NotImplementedError
    
    # ===== Geometry Tools =====
    # OPTIONAL: Kann auch außerhalb des Simulators implementiert werden
    @abstractmethod
    def geo_calculate_direction_and_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
        """Berrechnet Distanz(NM) und Richtung zwischen zwei Punkten (lat/lon) Returns (direction, distance)"""
        raise NotImplementedError
    