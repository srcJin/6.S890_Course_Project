#!/usr/bin/env python3
"""
Simple test script for the Urban Resilience SimCity Scale-Up Environment
"""
import sys
import os
import numpy as np

# Simple test without complex imports
def test_config():
    """Test that the config loads correctly"""
    print("Testing config loading...")
    
    # Load config directly
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    config_globals = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config_globals)
    
    # Check key configurations
    building_types = config_globals['BUILDING_TYPES']
    building_costs = config_globals['BUILDING_COSTS'] 
    grid_layout = config_globals['DEFAULT_GRID_LAYOUT']
    terrain = config_globals['DEFAULT_TERRAIN_ASSIGNMENT']
    infrastructure = config_globals['DEFAULT_INFRASTRUCTURE_ASSIGNMENT']
    
    print(f"✓ Building types ({len(building_types)}): {building_types}")
    print(f"✓ Grid layout shape: {np.array(grid_layout).shape}")
    print(f"✓ Terrain features: {len(terrain)} locations")
    print(f"✓ Infrastructure: {len(infrastructure)} buildings")
    
    # Display the default grid layout
    print("\nDefault 8x8 Grid Layout:")
    print("0=Buildable, 1=Terrain, 2=Infrastructure")
    for i, row in enumerate(grid_layout):
        print(f"Row {i}: {row}")
    
    print("\nTerrain assignments:")
    for pos, terrain_type in terrain.items():
        print(f"  Position {pos}: {terrain_type}")
    
    print("\nInfrastructure assignments:")
    for pos, infra_type in infrastructure.items():
        print(f"  Position {pos}: {infra_type}")
    
    print("\nBuilding costs and effects:")
    for building in building_types:
        cost = building_costs[building]
        print(f"  {building}: Cost=${cost['money']}, Rep={cost['reputation']}")
    
    print("\nConfig test completed successfully!")

if __name__ == "__main__":
    test_config()
