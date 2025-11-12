#!/usr/bin/env python3
"""
Einfacher Test für alle Simulator Adapter
Nutzt das SimulatorInterface um alle Implementierungen zu testen
"""

import sys
import os

# Füge den custom_envs Pfad hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), 'rl_zoo3', 'custom_envs'))

def test_simulator_interface(adapter_class):

    
    try:
        # 1. Erstelle Adapter Instanz
        print("  ✓ Erstelle Adapter...")
        adapter = adapter_class()
        
        # 2. Initialisierung
        print("  ✓ Initialisiere Simulator...")
        adapter.init(1.0)
        
        # 3. Reset Traffic
        print("  ✓ Reset Traffic...")
        adapter.traf_reset()
        
        # 4. Erstelle mehrere Flugzeuge
        print("  ✓ Erstelle Flugzeuge...")
        aircraft_list = [
            ("AC001", "B737", 250.0, 52.0, 4.0, 90.0, 10000.0),
            ("AC002", "A320", 280.0, 52.1, 4.1, 180.0, 12000.0),
            ("AC003", "B777", 300.0, 51.9, 3.9, 270.0, 15000.0)
        ]
        
        created_aircraft = []
        for acid, actype, spd, lat, lon, hdg, alt in aircraft_list:
            success = adapter.traf_create(acid, actype, spd, lat, lon, hdg, alt)
            if success:
                created_aircraft.append(acid)
                print(f"    ✓ {acid} erstellt")
            else:
                print(f"    ⚠️ {acid} konnte nicht erstellt werden")
        
        if not created_aircraft:
            print("    ⚠️ Keine Flugzeuge erstellt - teste trotzdem weiter")
        
        # 5. Simulation Step
        print("  ✓ Führe Simulation Step aus...")
        adapter.sim_step()
        
        # 6. Teste andere Funktionen mit erstellten Flugzeugen
        if created_aircraft:
            test_aircraft = created_aircraft[0]
            
            print(f"  ✓ Teste Funktionen mit {test_aircraft}...")
            
            # Position abfragen
            try:
                lat, lon, hdg, tas = adapter.traf_get_state(test_aircraft)
                print(f"    Position: lat={lat:.2f}, lon={lon:.2f}, hdg={hdg:.1f}°, tas={tas:.1f}kts")
            except Exception as e:
                print(f"    ⚠️ Position konnte nicht abgefragt werden: {e}")
            
            # Heading ändern
            try:
                adapter.traf_set_heading(test_aircraft, 45.0)
                print(f"    ✓ Heading auf 45° gesetzt")
            except Exception as e:
                print(f"    ⚠️ Heading konnte nicht gesetzt werden: {e}")
            
            # Speed ändern
            try:
                adapter.traf_set_speed(test_aircraft, 320.0)
                print(f"    ✓ Speed auf 320kts gesetzt")
            except Exception as e:
                print(f"    ⚠️ Speed konnte nicht gesetzt werden: {e}")
            
            # Altitude ändern
            try:
                adapter.traf_set_altitude(test_aircraft, 18000.0)
                print(f"    ✓ Altitude auf 18000ft gesetzt")
            except Exception as e:
                print(f"    ⚠️ Altitude konnte nicht gesetzt werden: {e}")
        
        # 7. Teste Geometrie-Funktionen
        print("  ✓ Teste Geometrie-Berechnung...")
        try:
            # Berechne Distanz zwischen Amsterdam und Berlin
            amsterdam_lat, amsterdam_lon = 52.3676, 4.9041
            berlin_lat, berlin_lon = 52.5200, 13.4050
            
            heading, distance = adapter.geo_calculate_direction_and_distance(
                amsterdam_lat, amsterdam_lon, berlin_lat, berlin_lon
            )
            print(f"    Amsterdam -> Berlin: {heading:.1f}°, {distance:.1f}NM")
        except Exception as e:
            print(f"    ⚠️ Geometrie-Berechnung fehlgeschlagen: {e}")
        
        # 8. Weiterer Simulation Step
        print("  ✓ Zweiter Simulation Step...")
        adapter.sim_step()
        
        print(f"  ✅ {adapter_name} erfolgreich getestet!")
        return True
        
    except Exception as e:
        print(f"  ❌ Fehler beim Testen von {adapter_name}: {e}")
        import traceback
        print(f"    Details: {traceback.format_exc()}")
        return False


def main():
    """Hauptfunktion zum Testen aller verfügbaren Adapter"""
    
    # Teste Blue Sky Adapter
    try:
        from blue_sky_adapter import Simulator
        if test_simulator_interface(Simulator):
            print("🎉 Blue Sky Adapter Test bestanden!")
    except ImportError as e:
        print(f"⚠️  Blue Sky Adapter nicht verfügbar: {e}")
    

if __name__ == '__main__':
    exit_code = main()