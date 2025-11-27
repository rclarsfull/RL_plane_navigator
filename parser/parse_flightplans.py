#!/usr/bin/env python3
"""
Script zum Parsen von Flugplandaten aus exercises.json
Extrahiert Lat, Lon, TAS, RFL Werte für jeden Flugplan, sortiert nach ETO
"""

import json
import re
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


def parse_latlon(latlon_str: str) -> Tuple[float, float]:
    """
    Parst einen LATLON String im Format "N50 23 44.18 E006 38 46.40"
    Gibt (lat, lon) als float-Tupel zurück
    """
    if not latlon_str:
        return None, None
    
    # Regex für das LATLON Format
    pattern = r'([NS])(\d+)\s+(\d+)\s+([\d.]+)\s+([EW])(\d+)\s+(\d+)\s+([\d.]+)'
    match = re.match(pattern, latlon_str)
    
    if not match:
        return None, None
    
    # Latitude parsen
    lat_dir, lat_deg, lat_min, lat_sec = match.groups()[:4]
    lat = float(lat_deg) + float(lat_min)/60 + float(lat_sec)/3600
    if lat_dir == 'S':
        lat = -lat
    
    # Longitude parsen
    lon_dir, lon_deg, lon_min, lon_sec = match.groups()[4:]
    lon = float(lon_deg) + float(lon_min)/60 + float(lon_sec)/3600
    if lon_dir == 'W':
        lon = -lon
    
    return lat, lon


def parse_tas(tas_str: str) -> int:
    """
    Parst TAS String im Format "N0454" und gibt die Geschwindigkeit als int zurück
    """
    if not tas_str:
        return None
    
    # Entferne den führenden Buchstaben (N, M, etc.) und konvertiere zu int
    match = re.match(r'[A-Z](\d+)', tas_str)
    if match:
        return int(match.group(1))
    return None


def parse_rfl(rfl_str: str) -> int:
    """
    Parst RFL String im Format "F370" und gibt die Flughöhe zurück
    """
    if not rfl_str:
        return None
    
    # Entferne den führenden Buchstaben (F, A, etc.) und konvertiere zu int
    match = re.match(r'[A-Z](\d+)', rfl_str)
    if match:
        return int(match.group(1))
    return None


def extract_flightplan_number(flightplan_ref: Any) -> int:
    """
    Extrahiert die Flugplannummer aus der Java BigDecimal Referenz
    """
    if isinstance(flightplan_ref, list) and len(flightplan_ref) == 2:
        if flightplan_ref[0] == "java.math.BigDecimal":
            return flightplan_ref[1]
    return None


def parse_exercises_json(file_path: str) -> Dict[int, List[Dict]]:
    """
    Parst die exercises.json Datei und extrahiert Flugplandaten
    
    Returns:
        Dict mit Flugplannummer als Key und Liste von Wegpunkten als Value
        Jeder Wegpunkt enthält: {'lat': float, 'lon': float, 'tas': int, 'rfl': int, 'eto': int}
        Die Listen sind nach ETO sortiert
    """
    
    #print("Lade JSON-Datei...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flightplans = defaultdict(list)
    
    def traverse_data(obj, level=0):
        """Rekursive Funktion zum Durchlaufen der verschachtelten Struktur"""
        if isinstance(obj, dict):
            # Prüfe ob das aktuelle Objekt Flugplandaten enthält
            if 'data' in obj and isinstance(obj['data'], dict):
                data_obj = obj['data']
                
                # Prüfe ob alle benötigten Felder vorhanden sind
                if all(key in data_obj for key in ['FLIGHTPLAN', 'ETO', 'TAS', 'LATLON', 'RFL']):
                    flightplan_num = extract_flightplan_number(data_obj['FLIGHTPLAN'])
                    
                    if flightplan_num is not None:
                        lat, lon = parse_latlon(data_obj['LATLON'])
                        tas = parse_tas(data_obj['TAS'])
                        rfl = parse_rfl(data_obj['RFL'])
                        eto = data_obj['ETO']
                        
                        # Nur hinzufügen wenn alle Werte geparst werden konnten
                        if lat is not None and lon is not None and tas is not None and rfl is not None:
                            waypoint = {
                                'lat': lat,
                                'lon': lon,
                                'tas': tas,
                                'rfl': rfl,
                                'eto': eto,
                                'fix': data_obj.get('FIX', ''),
                                'sector': data_obj.get('SECTOR', '')
                            }
                            flightplans[flightplan_num].append(waypoint)
            
            # Rekursiv durch alle Werte des Dictionaries
            for value in obj.values():
                traverse_data(value, level + 1)
                
        elif isinstance(obj, list):
            # Rekursiv durch alle Elemente der Liste
            for item in obj:
                traverse_data(item, level + 1)
    
    #print("Durchsuche Datenstruktur nach Flugplänen...")
    traverse_data(data)
    
    # Sortiere jeden Flugplan nach ETO
   # print("Sortiere Wegpunkte nach ETO...")
    for flightplan_num in flightplans:
        flightplans[flightplan_num].sort(key=lambda x: x['eto'])
    
    return dict(flightplans)


def get_all_waypoints_sorted_by_eto(file_path: str) -> List[Dict]:
    """
    Extrahiert alle Wegpunkte aus allen Flugplänen und sortiert sie nach ETO
    
    Returns:
        Liste von Dicts, jeder Dict enthält:
        {'lat': float, 'lon': float, 'tas': int, 'rfl': int, 'eto': int, 
         'fix': str, 'sector': str, 'flightplan': int}
        Sortiert nach ETO
    """
    flightplans = parse_exercises_json(file_path)
    
    all_waypoints = []
    
    # Sammle alle Wegpunkte aus allen Flugplänen
    for flightplan_num, waypoints in flightplans.items():
        for waypoint in waypoints:
            # Füge Flugplannummer hinzu
            waypoint_with_fp = waypoint.copy()
            waypoint_with_fp['flightplan'] = flightplan_num
            all_waypoints.append(waypoint_with_fp)
    
    # Sortiere alle Wegpunkte nach ETO
    all_waypoints.sort(key=lambda x: x['eto'])
    
    #print(f"Insgesamt {len(all_waypoints)} Wegpunkte aus {len(flightplans)} Flugplänen")
    
    return all_waypoints


def extract_exercise_info(data: Dict) -> Tuple[Optional[int], Optional[Dict]]:
    """
    Extrahiert Exercise-ID und Flugplandaten aus einer Flugplan-Datenstruktur
    
    Args:
        data: Dictionary mit Flugplandaten aus der JSON
    
    Returns:
        Tuple aus (exercise_id, flightplan_data) oder (None, None) falls nicht vorhanden
    """
    if not isinstance(data, dict):
        return None, None
    
    # Extrahiere Exercise ID (kann eine List mit BigDecimal Referenz sein)
    exercise_ref = data.get('EXERCISE')
    exercise_id = None
    
    if isinstance(exercise_ref, list) and len(exercise_ref) == 2:
        if exercise_ref[0] == "java.math.BigDecimal":
            exercise_id = exercise_ref[1]
    
    # Extrahiere relevante Flugplandaten
    flightplan_data = {
        'callsign': data.get('CALLSIGN', ''),
        'adep': data.get('ADEP', ''),  # Departure airport
        'ades': data.get('ADES', ''),  # Destination airport
        'rfl': parse_rfl(data.get('RFL', '')),  # Requested flight level
        'cfl': parse_rfl(data.get('CFL', '')),  # Current flight level
        'aircraft_type': data.get('AIRCRAFTTYPE', ''),
        'route': data.get('FIX', ''),
        'sector': data.get('SECTOR', ''),
    }
    
    return exercise_id, flightplan_data


def parse_all_exercises(file_path: str) -> Dict[int, Dict]:
    """
    Parst alle Exercises aus der JSON-Datei
    
    Returns:
        Dict mit Exercise-ID als Key und Flugplandaten als Value
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    exercises = {}
    
    def traverse_exercises(obj):
        """Rekursive Funktion zum Finden aller Exercises"""
        if isinstance(obj, dict):
            # Prüfe ob das aktuelle Objekt Flugplandaten mit Exercise enthält
            if 'data' in obj and isinstance(obj['data'], dict):
                data_obj = obj['data']
                
                # Prüfe ob Exercise Feld vorhanden ist
                if 'EXERCISE' in data_obj:
                    exercise_id, flightplan_data = extract_exercise_info(data_obj)
                    
                    if exercise_id is not None:
                        # Merge die Flugplandaten
                        exercises[exercise_id] = {
                            **flightplan_data,
                            'callsign': data_obj.get('CALLSIGN', ''),
                            'rfl': parse_rfl(data_obj.get('RFL', '')),
                            'aircraft_type': data_obj.get('AIRCRAFTTYPE', 'A320'),
                        }
            
            # Rekursiv durch alle Werte des Dictionaries
            for value in obj.values():
                traverse_exercises(value)
                
        elif isinstance(obj, list):
            # Rekursiv durch alle Elemente der Liste
            for item in obj:
                traverse_exercises(item)
    
    traverse_exercises(data)
    return exercises


def get_random_exercise(file_path: str) -> Tuple[int, Dict[int, List[Dict]]]:
    """
    Wählt ein zufälliges Exercise aus und gibt alle zugehörigen Flugpläne zurück
    
    Args:
        file_path: Pfad zur exercises.json Datei
    
    Returns:
        Tuple aus (exercise_id, dict_of_flightplans)
        dict_of_flightplans ist strukturiert wie parse_exercises_json()
    """
    # Parse alle Exercises
    exercises = parse_all_exercises(file_path)
    
    if not exercises:
        raise ValueError(f"Keine Exercises in {file_path} gefunden")
    
    # Wähle ein zufälliges Exercise
    exercise_id = random.choice(list(exercises.keys()))
    
    # Lade alle Flugpläne für dieses Exercise
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flightplans = defaultdict(list)
    
    def traverse_for_exercise(obj, target_exercise_id):
        """Rekursive Funktion zum Finden aller Flugpläne für ein spezifisches Exercise"""
        if isinstance(obj, dict):
            # Prüfe ob das aktuelle Objekt Flugplandaten mit dem richtigen Exercise enthält
            if 'data' in obj and isinstance(obj['data'], dict):
                data_obj = obj['data']
                
                # Prüfe ob Exercise Feld vorhanden und stimmt überein
                if 'EXERCISE' in data_obj:
                    ex_id, _ = extract_exercise_info(data_obj)
                    
                    if ex_id == target_exercise_id:
                        # Prüfe ob Flugplandaten vorhanden sind
                        if all(key in data_obj for key in ['FLIGHTPLAN', 'ETO', 'TAS', 'LATLON', 'RFL']):
                            flightplan_num = extract_flightplan_number(data_obj['FLIGHTPLAN'])
                            
                            if flightplan_num is not None:
                                lat, lon = parse_latlon(data_obj['LATLON'])
                                tas = parse_tas(data_obj['TAS'])
                                rfl = parse_rfl(data_obj['RFL'])
                                eto = data_obj['ETO']
                                
                                if lat is not None and lon is not None and tas is not None and rfl is not None:
                                    waypoint = {
                                        'lat': lat,
                                        'lon': lon,
                                        'tas': tas,
                                        'rfl': rfl,
                                        'eto': eto,
                                        'fix': data_obj.get('FIX', ''),
                                        'sector': data_obj.get('SECTOR', ''),
                                        'callsign': data_obj.get('CALLSIGN', ''),
                                        'aircraft_type': data_obj.get('AIRCRAFTTYPE', 'A320'),
                                    }
                                    flightplans[flightplan_num].append(waypoint)
            
            # Rekursiv durch alle Werte des Dictionaries
            for value in obj.values():
                traverse_for_exercise(value, target_exercise_id)
                
        elif isinstance(obj, list):
            # Rekursiv durch alle Elemente der Liste
            for item in obj:
                traverse_for_exercise(item, target_exercise_id)
    
    traverse_for_exercise(data, exercise_id)
    
    # Sortiere jeden Flugplan nach ETO
    for flightplan_num in flightplans:
        flightplans[flightplan_num].sort(key=lambda x: x['eto'])
    
    return exercise_id, dict(flightplans)


def select_flight_level_from_quartile(flightplans: Dict[int, List[Dict]], 
                                      quartile_min: float = 0.25,
                                      quartile_max: float = 0.75) -> int:
    """
    Wählt einen Flight Level aus dem angegebenen Quartil aus.
    
    Args:
        flightplans: Dictionary mit Flugplannummer als Key und Liste von Waypoints als Value
        quartile_min: Minimales Quartil (0-1, default 0.25 = 2. Quartil)
        quartile_max: Maximales Quartil (0-1, default 0.75 = 3. Quartil)
    
    Returns:
        Zufällig gewählter Flight Level aus dem Quartil
    """
    # Sammle alle RFL (Flight Levels)
    all_rfls = []
    for fp_id, waypoints in flightplans.items():
        for wp in waypoints:
            if 'rfl' in wp:
                all_rfls.append(wp['rfl'])
    
    if not all_rfls:
        return 350  # Default fallback
    
    all_rfls = sorted(set(all_rfls))
    
    # Berechne Quartil-Grenzen
    min_idx = int(len(all_rfls) * quartile_min)
    max_idx = int(len(all_rfls) * quartile_max)
    
    # Sicherstelle dass Indizes gültig sind
    min_idx = max(0, min_idx)
    max_idx = min(len(all_rfls) - 1, max_idx)
    
    if min_idx >= max_idx:
        return all_rfls[min_idx]
    
    # Wähle zufälligen RFL aus dem Quartil
    available_rfls = all_rfls[min_idx:max_idx + 1]
    return random.choice(available_rfls)


def filter_flightplans_by_time_window_and_flight_level(
    flightplans: Dict[int, List[Dict]], 
    window_duration_seconds: int = 3600,
    target_flight_level: Optional[int] = None,
    quartile_min: float = 0.25,
    quartile_max: float = 0.75,
    min_eto: Optional[int] = None,
    max_eto: Optional[int] = None,
    max_retries: int = 5
) -> Tuple[Dict[int, List[Dict]], int, int, int]:
    """
    Filtert Flugpläne basierend auf Zeitfenster und Flight Level.
    Wählt ein zufälliges Fenster und einen Flight Level aus dem Quartil.
    Behält nur Flugzeuge mit diesem Flight Level.
    Bei Bedarf werden mehrere Versuche unternommen um ein gültiges Fenster zu finden.
    
    Args:
        flightplans: Dictionary mit Flugplannummer als Key und Liste von Waypoints als Value
        window_duration_seconds: Dauer des Zeitfensters in Sekunden (default 3600 = 1 Stunde)
        target_flight_level: Zielflightlevel. Wenn None, wird eines aus dem Quartil gewählt
        quartile_min: Minimales Quartil (0-1, default 0.25 = 2. Quartil)
        quartile_max: Maximales Quartil (0-1, default 0.75 = 3. Quartil)
        min_eto: Minimale ETO zum Filtern (optional)
        max_eto: Maximale ETO zum Filtern (optional)
        max_retries: Maximale Anzahl von Versuchen (default 5)
    
    Returns:
        Tuple aus (filtered_flightplans, window_start_eto, window_end_eto, selected_flight_level)
        - filtered_flightplans: Gefilterte Flugpläne (nur mit target_flight_level)
        - window_start_eto: Start ETO des gewählten Fensters
        - window_end_eto: End ETO des gewählten Fensters
        - selected_flight_level: Der gewählte/verwendete Flight Level
    """
    if not flightplans:
        return {}, 0, window_duration_seconds, 350
    
    # Sammle alle ETOs
    all_etos = []
    for fp_id, waypoints in flightplans.items():
        for wp in waypoints:
            all_etos.append(wp['eto'])
    
    if not all_etos:
        return {}, 0, window_duration_seconds, 350
    
    all_etos = sorted(set(all_etos))
    min_global_eto = min(all_etos)
    max_global_eto = max(all_etos)
    
    # Versuche mehrmals ein gültiges Fenster zu finden
    for attempt in range(max_retries):
        # Wenn min_eto/max_eto nicht angegeben, ein zufälliges Fenster wählen
        if min_eto is None:
            # Wähle einen zufälligen Startpunkt, sodass das Fenster noch gültig ist
            available_start = max_global_eto - window_duration_seconds
            
            if available_start <= min_global_eto:
                # Fenster größer als verfügbare Daten - verwende alles
                window_start = min_global_eto
            else:
                # Wähle zufälligen Startpunkt
                window_start = random.randint(min_global_eto, available_start)
        else:
            window_start = min_eto
        
        window_end = window_start + window_duration_seconds
        
        # Wähle Flight Level aus Quartil (wenn nicht vorgegeben)
        if target_flight_level is None:
            target_flight_level = select_flight_level_from_quartile(
                flightplans, quartile_min, quartile_max
            )
        
        # Filtere Flugpläne nach Zeit und Flight Level
        filtered_flightplans = {}
        
        for fp_id, waypoints in flightplans.items():
            # Behalte nur Waypoints im Zeitfenster UND mit korrektem Flight Level
            filtered_waypoints = [
                wp for wp in waypoints 
                if (window_start <= wp['eto'] <= window_end and 
                    wp.get('rfl') == target_flight_level)
            ]
            
            # Nur hinzufügen wenn wenigstens einen Waypoint im Fenster vorhanden
            if filtered_waypoints:
                filtered_flightplans[fp_id] = filtered_waypoints
        
        # Wenn wir gültige Flugpläne gefunden haben, return
        if filtered_flightplans:
            return filtered_flightplans, window_start, window_end, target_flight_level
        
        # Sonst versuche es mit einem neuen Fenster/Flight Level
        target_flight_level = None  # Reset für nächsten Versuch
    
    # Fallback: Wenn nach allen Versuchen nichts gefunden, lockere die Bedingungen
    # Versuche nur nach Zeit zu filtern (ignoriere Flight Level Quartil)
    all_rfl_values = []
    for fp_id, waypoints in flightplans.items():
        for wp in waypoints:
            if 'rfl' in wp:
                all_rfl_values.append(wp['rfl'])
    
    if all_rfl_values:
        # Wähle einfach einen zufälligen Flight Level aus allen verfügbaren
        fallback_flight_level = random.choice(all_rfl_values)
        
        # Berechne ein Fenster, das diesen Flight Level enthält
        for fp_id, waypoints in flightplans.items():
            for wp in waypoints:
                if wp.get('rfl') == fallback_flight_level:
                    window_start = max(min_global_eto, wp['eto'] - window_duration_seconds // 2)
                    break
        
        window_end = window_start + window_duration_seconds
        
        filtered_flightplans = {}
        for fp_id, waypoints in flightplans.items():
            filtered_waypoints = [
                wp for wp in waypoints 
                if (window_start <= wp['eto'] <= window_end and 
                    wp.get('rfl') == fallback_flight_level)
            ]
            if filtered_waypoints:
                filtered_flightplans[fp_id] = filtered_waypoints
        
        if filtered_flightplans:
            return filtered_flightplans, window_start, window_end, fallback_flight_level
    
    # Letzter Ausweg: Gib einfach alle Flugpläne mit dem häufigsten Flight Level zurück
    rfl_counts = {}
    for fp_id, waypoints in flightplans.items():
        for wp in waypoints:
            rfl = wp.get('rfl')
            if rfl:
                rfl_counts[rfl] = rfl_counts.get(rfl, 0) + 1
    
    if rfl_counts:
        most_common_rfl = max(rfl_counts, key=rfl_counts.get)
        filtered_flightplans = {}
        for fp_id, waypoints in flightplans.items():
            filtered_waypoints = [wp for wp in waypoints if wp.get('rfl') == most_common_rfl]
            if filtered_waypoints:
                filtered_flightplans[fp_id] = filtered_waypoints
        
        if filtered_flightplans:
            return filtered_flightplans, min_global_eto, max_global_eto, most_common_rfl
    
    return {}, 0, window_duration_seconds, 350


def get_specific_exercise(file_path: str, exercise_id: int) -> Tuple[int, Dict[int, List[Dict]]]:
    """
    Lädt ein spezifisches Exercise anhand seiner ID
    
    Args:
        file_path: Pfad zur exercises.json Datei
        exercise_id: Die gewünschte Exercise ID
    
    Returns:
        Tuple aus (exercise_id, dict_of_flightplans)
        dict_of_flightplans ist strukturiert wie parse_exercises_json()
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flightplans = defaultdict(list)
    exercise_found = False
    
    def traverse_for_exercise(obj, target_exercise_id):
        """Rekursive Funktion zum Finden aller Flugpläne für ein spezifisches Exercise"""
        nonlocal exercise_found
        
        if isinstance(obj, dict):
            # Prüfe ob das aktuelle Objekt Flugplandaten mit dem richtigen Exercise enthält
            if 'data' in obj and isinstance(obj['data'], dict):
                data_obj = obj['data']
                
                # Prüfe ob Exercise Feld vorhanden und stimmt überein
                if 'EXERCISE' in data_obj:
                    ex_id, _ = extract_exercise_info(data_obj)
                    
                    if ex_id == target_exercise_id:
                        exercise_found = True
                        # Prüfe ob Flugplandaten vorhanden sind
                        if all(key in data_obj for key in ['FLIGHTPLAN', 'ETO', 'TAS', 'LATLON', 'RFL']):
                            flightplan_num = extract_flightplan_number(data_obj['FLIGHTPLAN'])
                            
                            if flightplan_num is not None:
                                lat, lon = parse_latlon(data_obj['LATLON'])
                                tas = parse_tas(data_obj['TAS'])
                                rfl = parse_rfl(data_obj['RFL'])
                                eto = data_obj['ETO']
                                
                                if lat is not None and lon is not None and tas is not None and rfl is not None:
                                    waypoint = {
                                        'lat': lat,
                                        'lon': lon,
                                        'tas': tas,
                                        'rfl': rfl,
                                        'eto': eto,
                                        'fix': data_obj.get('FIX', ''),
                                        'sector': data_obj.get('SECTOR', ''),
                                        'callsign': data_obj.get('CALLSIGN', ''),
                                        'aircraft_type': data_obj.get('AIRCRAFTTYPE', 'A320'),
                                    }
                                    flightplans[flightplan_num].append(waypoint)
            
            # Rekursiv durch alle Werte des Dictionaries
            for value in obj.values():
                traverse_for_exercise(value, target_exercise_id)
                
        elif isinstance(obj, list):
            # Rekursiv durch alle Elemente der Liste
            for item in obj:
                traverse_for_exercise(item, target_exercise_id)
    
    traverse_for_exercise(data, exercise_id)
    
    if not exercise_found:
        raise ValueError(f"Exercise mit ID {exercise_id} nicht gefunden in {file_path}")
    
    # Sortiere jeden Flugplan nach ETO
    for flightplan_num in flightplans:
        flightplans[flightplan_num].sort(key=lambda x: x['eto'])
    
    return exercise_id, dict(flightplans)


def main():
    """Hauptfunktion zum Ausführen des Scripts"""
    
    file_path = "exercises.json"
    
    try:
        # Parse die JSON-Datei
        flightplans = parse_exercises_json(file_path)
        
        print(f"\nGefundene Flugpläne: {len(flightplans)}")
        
        # Zeige Statistiken für jeden Flugplan
        for flightplan_num, waypoints in flightplans.items():
            print(f"\nFlugplan {flightplan_num}:")
            print(f"  Anzahl Wegpunkte: {len(waypoints)}")
            
            if waypoints:
                print(f"  ETO Bereich: {waypoints[0]['eto']} - {waypoints[-1]['eto']}")
                print(f"  Beispiel Wegpunkt:")
                wp = waypoints[0]
                print(f"    Lat: {wp['lat']:.6f}, Lon: {wp['lon']:.6f}")
                print(f"    TAS: {wp['tas']}, RFL: {wp['rfl']}, ETO: {wp['eto']}")
                print(f"    Fix: {wp['fix']}, Sector: {wp['sector']}")
        
        # Beispiel: Zugriff auf die Daten
        print(f"\n--- Beispiel für Datenzugriff ---")
        if flightplans:
            first_flightplan = next(iter(flightplans.keys()))
            waypoints = flightplans[first_flightplan]
            
            print(f"Flugplan {first_flightplan} hat {len(waypoints)} Wegpunkte:")
            
            # Extrahiere Arrays für Lat, Lon, TAS, RFL
            lats = [wp['lat'] for wp in waypoints]
            lons = [wp['lon'] for wp in waypoints]
            tas_values = [wp['tas'] for wp in waypoints]
            rfl_values = [wp['rfl'] for wp in waypoints]
            eto_values = [wp['eto'] for wp in waypoints]
            
            print(f"  Latitudes: {lats[:3]}... (erste 3)")
            print(f"  Longitudes: {lons[:3]}... (erste 3)")
            print(f"  TAS: {tas_values[:3]}... (erste 3)")
            print(f"  RFL: {rfl_values[:3]}... (erste 3)")
            print(f"  ETO: {eto_values[:3]}... (erste 3)")
        
        # Teste die neue Funktion
        print(f"\n--- Test: Alle Wegpunkte sortiert nach ETO ---")
        all_waypoints = get_all_waypoints_sorted_by_eto(file_path)
        
        if all_waypoints:
            print(f"Erste 3 Wegpunkte (nach ETO sortiert):")
            for i, wp in enumerate(all_waypoints[:3]):
                print(f"  {i+1}. ETO: {wp['eto']}, Flugplan: {wp['flightplan']}, "
                      f"Lat: {wp['lat']:.4f}, Lon: {wp['lon']:.4f}, "
                      f"TAS: {wp['tas']}, RFL: {wp['rfl']}, Fix: {wp['fix']}")
            
            print(f"\nLetzte 3 Wegpunkte:")
            for i, wp in enumerate(all_waypoints[-3:], len(all_waypoints)-2):
                print(f"  {i}. ETO: {wp['eto']}, Flugplan: {wp['flightplan']}, "
                      f"Lat: {wp['lat']:.4f}, Lon: {wp['lon']:.4f}, "
                      f"TAS: {wp['tas']}, RFL: {wp['rfl']}, Fix: {wp['fix']}")
        
        return flightplans
        
    except Exception as e:
        print(f"Fehler beim Parsen der Datei: {e}")
        return {}


if __name__ == "__main__":
    flightplans = main()