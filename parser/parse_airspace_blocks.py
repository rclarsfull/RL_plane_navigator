#!/usr/bin/env python3
"""
Parser for Airspace blocks from ACS_Rating_2020.st file.
Parses ASB (Airspace Block) entries with their boundary points.
"""

import re
from typing import Dict, List, Tuple, Set


def parse_coordinate_string(coord_str: str) -> Tuple[float, float]:
    """
    Parse coordinate string like 'N50 11 13.00 E010 15 18.00' to (lat, lon) in decimal degrees.
    
    Args:
        coord_str: Coordinate string in format 'N50 11 13.00 E010 15 18.00'
        
    Returns:
        Tuple of (latitude, longitude) in decimal degrees
    """
    # Pattern to match coordinates like N50 11 13.00 E010 15 18.00
    pattern = r'([NS])(\d+)\s+(\d+)\s+(\d+\.\d+)\s+([EW])(\d+)\s+(\d+)\s+(\d+\.\d+)'
    match = re.match(pattern, coord_str.strip())
    
    if not match:
        raise ValueError(f"Invalid coordinate format: {coord_str}")
    
    # Parse latitude
    lat_dir, lat_deg, lat_min, lat_sec = match.groups()[:4]
    latitude = int(lat_deg) + int(lat_min)/60 + float(lat_sec)/3600
    if lat_dir == 'S':
        latitude = -latitude
    
    # Parse longitude
    lon_dir, lon_deg, lon_min, lon_sec = match.groups()[4:]
    longitude = int(lon_deg) + int(lon_min)/60 + float(lon_sec)/3600
    if lon_dir == 'W':
        longitude = -longitude
    
    return latitude, longitude


def parse_flight_level(fl_str: str) -> int:
    """
    Parse flight level string like 'F365' or 'A000' to integer.
    
    Args:
        fl_str: Flight level string
        
    Returns:
        Flight level as integer (0 for A000, actual level for F### format)
    """
    if fl_str.startswith('A'):
        return 0  # A000 means surface level
    elif fl_str.startswith('F'):
        return int(fl_str[1:])
    else:
        raise ValueError(f"Unknown flight level format: {fl_str}")


def parse_airspace_blocks(file_path: str, flight_level: int = None) -> Dict[str, Dict[str, any]]:
    """
    Parse airspace blocks from the ACS_Rating_2020.st file.
    
    Args:
        file_path: Path to the ACS_Rating_2020.st file
        flight_level: Optional flight level filter. If provided, only returns blocks that contain this FL
        
    Returns:
        Dictionary with structure: {BlockName: {'minFL': int, 'maxFL': int, 'coords': set of (lat, lon) tuples}}
    """
    airspace_blocks = {}
    current_block = None
    in_airspace_section = False
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Check if we're in the airspace blocks section
            if line == "###### Airspace Blocks ######":
                in_airspace_section = True
                continue
            
            # Skip lines before airspace section
            if not in_airspace_section:
                continue
            
            # Stop if we reach another section
            if line.startswith("######") and "Airspace Blocks" not in line:
                break
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse ASB (Airspace Block) line
            if line.startswith('ASB'):
                parts = line.split()
                if len(parts) >= 5:
                    # ASB 1 ERL1 F365 F315
                    block_id = parts[1]
                    block_name = parts[2]
                    max_fl_str = parts[3]  # First FL is usually the upper limit
                    min_fl_str = parts[4]  # Second FL is usually the lower limit
                    
                    current_block = {
                        'minFL': parse_flight_level(min_fl_str),
                        'maxFL': parse_flight_level(max_fl_str),
                        'coords': set()
                    }
                    
                    # Filter by flight level if specified
                    if flight_level is not None:
                        min_fl = current_block['minFL']
                        max_fl = current_block['maxFL']
                        # Check if the specified flight level is within the block's range
                        if not (min_fl <= flight_level <= max_fl):
                            current_block = None  # Skip this block
                            continue
                    
                    airspace_blocks[block_name] = current_block
            
            # Parse BP (Boundary Point) line
            elif line.startswith('BP') and current_block is not None:
                # BP N50 11 13.00 E010 15 18.00
                coord_part = line[2:].strip()  # Remove 'BP' prefix
                try:
                    lat, lon = parse_coordinate_string(coord_part)
                    current_block['coords'].add((lat, lon))
                except ValueError as e:
                    print(f"Warning: Could not parse coordinate: {coord_part} - {e}")
    
    return airspace_blocks


def print_airspace_summary(airspace_blocks: Dict[str, Dict[str, any]]):
    """Print a summary of parsed airspace blocks."""
    print(f"Parsed {len(airspace_blocks)} airspace blocks:")
    print("-" * 60)
    
    for block_name, block_data in airspace_blocks.items():
        print(f"Block: {block_name}")
        print(f"  Flight Levels: {block_data['minFL']} - {block_data['maxFL']}")
        print(f"  Boundary Points: {len(block_data['coords'])}")
        
        # Show first few coordinates as example
        coords_list = list(block_data['coords'])[:3]
        for i, (lat, lon) in enumerate(coords_list):
            print(f"    Point {i+1}: {lat:.6f}°N, {lon:.6f}°E")
        
        if len(block_data['coords']) > 3:
            print(f"    ... and {len(block_data['coords']) - 3} more points")
        print()


if __name__ == "__main__":
    # Example usage
    file_path = "ACS_Rating_2020.st"
    
    try:
        # Parse all airspace blocks
        print("=== All Airspace Blocks ===")
        airspace_blocks = parse_airspace_blocks(file_path)
        print_airspace_summary(airspace_blocks)
        
        # Parse only blocks that contain FL 300
        print("\n=== Airspace Blocks containing FL 300 ===")
        fl300_blocks = parse_airspace_blocks(file_path, flight_level=300)
        print_airspace_summary(fl300_blocks)
        
        # Parse only blocks that contain FL 100 (lower altitude)
        print("\n=== Airspace Blocks containing FL 100 ===")
        fl100_blocks = parse_airspace_blocks(file_path, flight_level=100)
        print_airspace_summary(fl100_blocks)
        
        # Example: Access specific block data
        if "ERL1" in airspace_blocks:
            erl1 = airspace_blocks["ERL1"]
            print(f"\nERL1 Block Details:")
            print(f"Min FL: {erl1['minFL']}")
            print(f"Max FL: {erl1['maxFL']}")
            print(f"Coordinates: {erl1['coords']}")
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error parsing file: {e}")